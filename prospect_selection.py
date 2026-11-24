"""Reproducible Hillstrom RCT analysis for budget-constrained prospect selection."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "hillstrom.csv"
DEFAULT_OUTPUT = ROOT / "outputs"
DATA_URL = (
    "http://www.minethatdata.com/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)
NUMERIC_FEATURES = ["recency", "history", "mens", "womens", "newbie"]
CATEGORICAL_FEATURES = ["history_segment", "zip_code", "channel"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REQUIRED_COLUMNS = {
    *FEATURES,
    "segment",
    "visit",
    "conversion",
    "spend",
}
METHODS = [
    ("Random", "Random selection"),
    ("RFM", "RFM history"),
    ("Calibrated EV", "Calibrated expected value"),
    ("X-learner", "X-learner uplift"),
]
NAVY = "#10233F"
BLUE = "#3B82F6"
TEAL = "#12B8A6"
ORANGE = "#F59E0B"
MUTED = "#65758B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to Hillstrom CSV")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=1500)
    parser.add_argument("--unit-cost", type=float, default=5.0)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--download", action="store_true", help="Download/replace the public CSV")
    return parser.parse_args()


def prepare_data(path: Path, download: bool) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    if download:
        try:
            urllib.request.urlretrieve(DATA_URL, path)
        except Exception as exc:
            raise RuntimeError(f"Could not download the public dataset from {DATA_URL}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}. Run with --download or provide --data.")

    df = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    df = df.loc[df["segment"].isin(["Womens E-Mail", "No E-Mail"])].copy()
    df["treatment"] = (df["segment"] == "Womens E-Mail").astype("int8")

    if df.empty or set(df["segment"].unique()) != {"Womens E-Mail", "No E-Mail"}:
        raise ValueError("Both expected randomized arms must be present in the dataset.")
    if df[FEATURES + ["visit", "conversion", "spend"]].isna().any().any():
        raise ValueError("Selected features and outcomes must not contain missing values.")
    for column in ("visit", "conversion"):
        if not set(df[column].unique()).issubset({0, 1}):
            raise ValueError(f"{column} must be a binary 0/1 outcome.")
    if (df["spend"] < 0).any():
        raise ValueError("Spend must be non-negative.")
    return df.reset_index(drop=True)


def make_preprocessor() -> OneHotEncoder:
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def make_classifier(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=120,
        max_leaf_nodes=15,
        learning_rate=0.07,
        l2_regularization=1.0,
        random_state=seed,
    )


def covariate_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Return absolute standardized differences for numeric and one-hot features."""
    treated = df.loc[df["treatment"] == 1]
    control = df.loc[df["treatment"] == 0]
    columns: dict[str, pd.Series] = {
        column: df[column].astype(float) for column in NUMERIC_FEATURES
    }
    for column in CATEGORICAL_FEATURES:
        for value in sorted(df[column].astype(str).unique()):
            columns[f"{column}={value}"] = (df[column].astype(str) == value).astype(float)

    rows = []
    for name, values in columns.items():
        a = values.loc[treated.index]
        b = values.loc[control.index]
        pooled_sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        difference = float(a.mean() - b.mean())
        smd = difference / pooled_sd if pooled_sd > 0 else 0.0
        rows.append(
            {
                "feature": name,
                "treated_mean": float(a.mean()),
                "control_mean": float(b.mean()),
                "smd": float(smd),
                "abs_smd": abs(float(smd)),
            }
        )
    balance = pd.DataFrame(rows)
    balance["feature_field"] = balance["feature"].str.split("=", n=1).str[0]
    return balance.sort_values("abs_smd", ascending=False).reset_index(drop=True)


def transform_features(encoder: OneHotEncoder, train: pd.DataFrame, test: pd.DataFrame):
    cat_train = encoder.fit_transform(train[CATEGORICAL_FEATURES].astype(str))
    cat_test = encoder.transform(test[CATEGORICAL_FEATURES].astype(str))
    x_train = np.column_stack([train[NUMERIC_FEATURES].to_numpy(dtype=float), cat_train])
    x_test = np.column_stack([test[NUMERIC_FEATURES].to_numpy(dtype=float), cat_test])
    return x_train, x_test


def qini_curve(scores: np.ndarray, treatment: np.ndarray, outcome: np.ndarray):
    order = np.argsort(-scores, kind="stable")
    t = treatment[order]
    y = outcome[order]
    cumulative_treated = np.cumsum(t == 1)
    cumulative_control = np.cumsum(t == 0)
    cumulative_y_treated = np.cumsum((t == 1) & (y == 1))
    cumulative_y_control = np.cumsum((t == 0) & (y == 1))
    scaled_control = np.divide(
        cumulative_y_control * cumulative_treated,
        cumulative_control,
        out=np.zeros(len(scores), dtype=float),
        where=cumulative_control > 0,
    )
    return np.arange(1, len(scores) + 1), cumulative_y_treated - scaled_control


def interval_difference(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) - 1.96 * se), float((a.mean() - b.mean()) + 1.96 * se)


def bootstrap_difference_ci(
    treated: np.ndarray, control: np.ndarray, reps: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    boot_t = np.empty(reps, dtype=float)
    boot_c = np.empty(reps, dtype=float)
    batch = 250
    start = 0
    while start < reps:
        size = min(batch, reps - start)
        idx_t = rng.integers(0, len(treated), size=(size, len(treated)))
        idx_c = rng.integers(0, len(control), size=(size, len(control)))
        boot_t[start : start + size] = treated[idx_t].mean(axis=1)
        boot_c[start : start + size] = control[idx_c].mean(axis=1)
        start += size
    delta = boot_t - boot_c
    low, high = np.quantile(delta, [0.025, 0.975])
    return float(low), float(high)


def evaluate_shortlist(
    scores: np.ndarray,
    name: str,
    test: pd.DataFrame,
    k: int,
    unit_cost: float,
    bootstrap_reps: int,
    seed: int,
) -> tuple[dict, np.ndarray]:
    chosen = np.argsort(-scores, kind="stable")[:k]
    selected = test.iloc[chosen]
    t = selected["treatment"].to_numpy()
    treated = selected.loc[selected["treatment"] == 1]
    control = selected.loc[selected["treatment"] == 0]
    if min(len(treated), len(control)) < 2:
        raise ValueError(f"{name} shortlist has too few customers in one randomized arm.")

    visit_t = treated["visit"].to_numpy(dtype=float)
    visit_c = control["visit"].to_numpy(dtype=float)
    spend_t = treated["spend"].to_numpy(dtype=float)
    spend_c = control["spend"].to_numpy(dtype=float)
    visit_lift = float(visit_t.mean() - visit_c.mean())
    spend_lift = float(spend_t.mean() - spend_c.mean())
    spend_ci = bootstrap_difference_ci(spend_t, spend_c, bootstrap_reps, seed)
    visit_ci = interval_difference(visit_t, visit_c)
    return (
        {
            "method": name,
            "selected_n": int(k),
            "treated_n": int(len(treated)),
            "control_n": int(len(control)),
            "treated_visit_rate": float(visit_t.mean()),
            "control_visit_rate": float(visit_c.mean()),
            "visit_lift": visit_lift,
            "visit_ci_low": visit_ci[0],
            "visit_ci_high": visit_ci[1],
            "treated_spend_per_customer": float(spend_t.mean()),
            "control_spend_per_customer": float(spend_c.mean()),
            "spend_lift_per_customer": spend_lift,
            "spend_ci_low": spend_ci[0],
            "spend_ci_high": spend_ci[1],
            "incremental_spend": float(k * spend_lift),
            "net_contribution": float(k * (spend_lift - unit_cost)),
            "net_contribution_ci_low": float(k * (spend_ci[0] - unit_cost)),
            "net_contribution_ci_high": float(k * (spend_ci[1] - unit_cost)),
        },
        chosen,
    )


def profile_shortlists(test: pd.DataFrame, chosen: dict[str, np.ndarray]) -> pd.DataFrame:
    profile_rows = []
    for name, indices in chosen.items():
        selected = test.iloc[indices]
        profile_rows.append(
            {
                "method": name,
                "n": int(len(selected)),
                "history_mean": float(selected["history"].mean()),
                "recency_mean_months": float(selected["recency"].mean()),
                "womens_buyer_pct": float(selected["womens"].mean() * 100),
                "multichannel_pct": float((selected["channel"] == "Multichannel").mean() * 100),
            }
        )
    return pd.DataFrame(profile_rows)


def save_calibration_plot(
    y_true: np.ndarray, raw: np.ndarray, calibrated: np.ndarray, path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 5.0), dpi=180)
    ax.plot([0, 0.35], [0, 0.35], "--", color="#94A3B8", linewidth=1.5, label="Perfect calibration")
    for values, label, color in [
        (raw, "Uncalibrated", BLUE),
        (calibrated, "Isotonic", TEAL),
    ]:
        observed, predicted = calibration_curve(
            y_true, values, n_bins=10, strategy="quantile"
        )
        brier = brier_score_loss(y_true, values)
        ax.plot(
            predicted,
            observed,
            marker="o",
            linewidth=2.2,
            markersize=5,
            color=color,
            label=f"{label} · Brier {brier:.4f}",
        )
    ax.set(xlim=(0, 0.35), ylim=(0, 0.35), xlabel="Mean predicted visit probability", ylabel="Observed visit rate")
    ax.set_title("Probability calibration · treated holdout", loc="left", color=NAVY, weight="bold", pad=12)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_qini_plot(
    scores: dict[str, np.ndarray],
    treatment: np.ndarray,
    outcome: np.ndarray,
    k: int,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 4.7), dpi=180)
    colors = {"RFM history": MUTED, "Calibrated EV": BLUE, "X-learner uplift": TEAL}
    for label, score in scores.items():
        xs, ys = qini_curve(score, treatment, outcome)
        ax.plot(xs, ys, linewidth=2.2, color=colors[label], label=label)
    xs_random, ys_random = qini_curve(
        np.random.default_rng(42).random(len(treatment)), treatment, outcome
    )
    ax.plot(
        [0, len(treatment)],
        [0, ys_random[-1]],
        linestyle="--",
        linewidth=1.6,
        color="#9AA7B7",
        label="Random-targeting reference",
    )
    ax.axvline(k, color=ORANGE, linestyle=":", linewidth=2, label=f"Budget cap · K={k:,}")
    ax.axhline(0, color="#D8DEE8", linewidth=0.8)
    ax.set(xlabel="Customers targeted (sorted by training-only score)", ylabel="Cumulative incremental visits")
    ax.set_title("Ranking quality across campaign depth", loc="left", color=NAVY, weight="bold", pad=12)
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_balance_plot(balance: pd.DataFrame, path: Path) -> None:
    group_rows = []
    for _, group in balance.groupby("feature_field", sort=True):
        row = group.loc[group["abs_smd"].idxmax()]
        group_rows.append({"feature": row["feature_field"], "smd": row["smd"], "abs_smd": row["abs_smd"]})
    selected = pd.DataFrame(group_rows).sort_values("smd")
    fig, ax = plt.subplots(figsize=(7.5, 4.1), dpi=180)
    colors = [TEAL if value < 0.1 else ORANGE for value in selected["abs_smd"]]
    ax.barh(selected["feature"], selected["smd"], color=colors, height=0.68)
    ax.axvline(0, color="#64748B", linewidth=0.8)
    ax.axvline(-0.1, color="#94A3B8", linewidth=1, linestyle="--")
    ax.axvline(0.1, color="#94A3B8", linewidth=1, linestyle="--", label="±0.10 reference")
    ax.set_xlabel("Standardized mean difference · treatment minus control")
    ax.set_title("Pre-campaign covariate balance", loc="left", color=NAVY, weight="bold", pad=10)
    ax.grid(axis="x", alpha=0.16)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.k <= 0 or args.bootstrap_reps < 100 or args.unit_cost < 0:
        raise ValueError("--k must be positive, --bootstrap-reps at least 100, and --unit-cost non-negative.")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    df = prepare_data(args.data.resolve(), args.download)
    if args.k >= len(df):
        raise ValueError(f"--k must be smaller than the two-arm cohort size ({len(df):,}).")
    balance = covariate_balance(df)
    strat_key = df["treatment"].astype(str) + "_" + df["visit"].astype(str)
    train, test = train_test_split(
        df, test_size=0.30, random_state=args.seed, stratify=strat_key
    )
    encoder = make_preprocessor()
    x_train, x_test = transform_features(encoder, train, test)
    t_train = train["treatment"].to_numpy()
    t_test = test["treatment"].to_numpy()
    y_train = train["visit"].to_numpy(dtype=int)
    y_test = test["visit"].to_numpy(dtype=int)

    x_train_t, y_train_t = x_train[t_train == 1], y_train[t_train == 1]
    x_train_c, y_train_c = x_train[t_train == 0], y_train[t_train == 0]
    x_test_t = x_test[t_test == 1]
    y_test_t = y_test[t_test == 1]

    raw_treated_model = make_classifier(args.seed).fit(x_train_t, y_train_t)
    raw_test_treated = raw_treated_model.predict_proba(x_test_t)[:, 1]
    calibrated_treated = CalibratedClassifierCV(
        estimator=make_classifier(args.seed), method="isotonic", cv=5
    ).fit(x_train_t, y_train_t)
    calibrated_control = CalibratedClassifierCV(
        estimator=make_classifier(args.seed), method="isotonic", cv=5
    ).fit(x_train_c, y_train_c)
    p1_test = calibrated_treated.predict_proba(x_test)[:, 1]
    p0_test = calibrated_control.predict_proba(x_test)[:, 1]

    # Each pseudo-outcome uses the opposite arm's model, never a model trained
    # on the customer's own observed outcome.
    p0_for_treated = calibrated_control.predict_proba(x_train_t)[:, 1]
    p1_for_control = calibrated_treated.predict_proba(x_train_c)[:, 1]
    pseudo_treated = y_train_t - p0_for_treated
    pseudo_control = p1_for_control - y_train_c
    tau1 = HistGradientBoostingRegressor(
        max_iter=100, max_leaf_nodes=10, learning_rate=0.05, l2_regularization=2.0, random_state=args.seed
    ).fit(x_train_t, pseudo_treated)
    tau0 = HistGradientBoostingRegressor(
        max_iter=100, max_leaf_nodes=10, learning_rate=0.05, l2_regularization=2.0, random_state=args.seed + 1
    ).fit(x_train_c, pseudo_control)
    propensity = float(t_train.mean())
    tau_x = propensity * tau0.predict(x_test) + (1 - propensity) * tau1.predict(x_test)

    scores = {
        "Random": np.random.default_rng(args.seed).random(len(test)),
        "RFM": test["history"].to_numpy(dtype=float),
        "Calibrated EV": p1_test * np.log1p(test["history"].to_numpy(dtype=float)),
        "X-learner": tau_x,
    }
    metrics = []
    chosen: dict[str, np.ndarray] = {}
    for offset, (key, label) in enumerate(METHODS):
        result, indices = evaluate_shortlist(
            scores[key],
            label,
            test,
            args.k,
            args.unit_cost,
            args.bootstrap_reps,
            args.seed + 100 + offset,
        )
        metrics.append(result)
        chosen[key] = indices

    profile = profile_shortlists(test, chosen)
    overlap = len(set(chosen["Calibrated EV"]).intersection(chosen["X-learner"]))
    jaccard = overlap / len(set(chosen["Calibrated EV"]).union(chosen["X-learner"]))
    treated_rate = float(df.loc[df["treatment"] == 1, "visit"].mean())
    control_rate = float(df.loc[df["treatment"] == 0, "visit"].mean())
    treated_spend = float(df.loc[df["treatment"] == 1, "spend"].mean())
    control_spend = float(df.loc[df["treatment"] == 0, "spend"].mean())
    balance_by_field = balance.groupby("feature_field")["abs_smd"].max()

    results = {
        "data": {
            "source": DATA_URL,
            "source_rows": int(len(pd.read_csv(args.data))),
            "two_arm_n": int(len(df)),
            "train_n": int(len(train)),
            "test_n": int(len(test)),
            "treated_n": int((df["treatment"] == 1).sum()),
            "control_n": int((df["treatment"] == 0).sum()),
            "treatment_rate": float(df["treatment"].mean()),
            "outcome_definition": "Website visit within two weeks of the randomized email campaign",
        },
        "design": {
            "seed": args.seed,
            "test_fraction": 0.30,
            "k": args.k,
            "unit_cost": args.unit_cost,
            "unit_cost_is_scenario_assumption": True,
            "max_absolute_smd": float(balance_by_field.max()),
            "balance_feature_count": int(len(balance_by_field)),
            "balance_fields_under_010": int((balance_by_field < 0.10).sum()),
            "features": FEATURES,
        },
        "overall": {
            "control_visit_rate": control_rate,
            "treated_visit_rate": treated_rate,
            "visit_lift": treated_rate - control_rate,
            "control_spend_per_customer": control_spend,
            "treated_spend_per_customer": treated_spend,
            "spend_lift_per_customer": treated_spend - control_spend,
        },
        "calibration": {
            "brier_uncalibrated": float(brier_score_loss(y_test_t, raw_test_treated)),
            "brier_isotonic": float(
                brier_score_loss(y_test_t, calibrated_treated.predict_proba(x_test_t)[:, 1])
            ),
        },
        "shortlists": metrics,
        "overlap": {
            "calibrated_ev_x_xlearner_n": int(overlap),
            "calibrated_ev_x_xlearner_pct_of_k": float(overlap / args.k),
            "jaccard": float(jaccard),
        },
        "profiles": profile.to_dict(orient="records"),
        "interpretation": {
            "economic_metric": "Incremental gross spend less assumed contact cost; not accounting profit.",
            "source_intervention": "Randomized email. A catalog/voucher campaign is an external-validity assumption.",
            "uncertainty": "Visit intervals use a normal approximation; spend intervals use a stratified percentile bootstrap.",
        },
    }

    balance.to_csv(output / "covariate_balance.csv", index=False)
    pd.DataFrame(metrics).to_csv(output / "shortlist_metrics.csv", index=False)
    profile.to_csv(output / "shortlist_profiles.csv", index=False)
    with (output / "results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    save_calibration_plot(
        y_test_t,
        raw_test_treated,
        calibrated_treated.predict_proba(x_test_t)[:, 1],
        output / "calibration.png",
    )
    save_qini_plot(
        {
            "RFM history": scores["RFM"],
            "Calibrated EV": scores["Calibrated EV"],
            "X-learner uplift": scores["X-learner"],
        },
        t_test,
        y_test,
        args.k,
        output / "qini.png",
    )
    save_balance_plot(balance, output / "balance.png")

    print(f"Two-arm randomized cohort: {len(df):,} | Train: {len(train):,} | Holdout: {len(test):,}")
    print(
        f"Overall visit rates: control {control_rate:.2%} | treated {treated_rate:.2%} "
        f"| lift {(treated_rate - control_rate):+.2%}"
    )
    print(
        f"Maximum absolute pre-campaign SMD: {balance['abs_smd'].max():.3f} "
        f"({(balance_by_field < 0.10).sum()}/{len(balance_by_field)} feature fields below 0.10)"
    )
    print(
        f"Brier score, treated holdout: raw {results['calibration']['brier_uncalibrated']:.5f} "
        f"| isotonic {results['calibration']['brier_isotonic']:.5f}"
    )
    print(f"\nTop-{args.k:,} randomized holdout comparison; unit contact cost = ${args.unit_cost:.2f}")
    printable = pd.DataFrame(metrics)[
        ["method", "treated_n", "control_n", "visit_lift", "spend_lift_per_customer", "spend_ci_low", "spend_ci_high", "net_contribution"]
    ].copy()
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(printable.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
    print(f"\nTop-{args.k:,} overlap (EV vs X-learner): {overlap} ({overlap / args.k:.1%}); Jaccard {jaccard:.1%}")
    print(f"Analysis outputs: {output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
