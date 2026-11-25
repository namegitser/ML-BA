---
geometry: margin=0.8in
fontsize: 10pt
---

# Team Project: Prospect Selection

**Machine Learning for Business Analytics, Plaksha MS-AI, Monsoon 2026.** Draft for release Mon 16 Nov. Weight: 35% of the course grade.

## The problem

An investment bank can arrange only 50 investor meetings for a company preparing to go public. A supplier cannot prepare proposals for every contract on offer. An organisation looking for partners, locations or markets must decide where to spend limited time and resources. These are one problem: **an organisation has more opportunities than it can pursue, and which ones it chooses matters.** Your team will choose a real setting where this problem arises and build a data-driven way of making the choice. You are not given a dataset, a prediction target or a method; finding a workable formulation, constructing the evidence and deciding how to evaluate it are part of the assignment. You may use AI tools freely.

## What the project must do

1. **Establish the decision.** Describe the kind of organisation, the opportunities it chooses among, and the limit on how many it can pursue (K); you need not name a specific organisation. Say what a good decision would achieve, and produce a ranking for a clearly defined decision situation.
2. **Build the dataset and the formulation.** Assemble real data and explain what each observation represents. Choose an approach that fits the decision (prediction, preference learning, similarity, ranking or something else) and say why; simple methods are fine where they fit.
3. **Show whether it works.** Evaluate your ranking using evidence not used to fit or choose your method, and compare it with a meaningful simple baseline; beating random selection is not enough. Explain what your evaluation establishes and what it cannot. The decision should be a real one, relevant in its setting; it need not be a fully reconstructed historical decision. For predictions about future outcomes, use only information available before those outcomes.
4. **Compare two approaches.** Compare two defensible ways of choosing K, show how the shortlists differ and explain why. The approaches may be technical variants; a substantive trade-off between them is not required, and no recommendation to the organisation is required. The simple baseline may be one of the two approaches.

Choosing a feasible problem is your responsibility. Before committing to a direction, establish that the data is accessible, the opportunities can be identified, and there is an observable outcome against which you can evaluate your choices. A convincing business idea without usable data is not sufficient.

A project is not successful merely because it produces predictions. You must demonstrate how they could have been evaluated against observable outcomes. If your data cannot support this, change the question or explain precisely what can be established.

## Four guiding questions

These are guides for your work and your presentation; your project may not satisfy all of them.

- What exactly does your model predict or rank? Is the outcome you observe a good measure of what the organisation values, and what is the gap between the two?
- Could a simple rule (the most active, the largest, the most established) achieve much of your result? What does your approach add, and how do you show it?
- What was known at the decision date? Which outcomes can you observe and which not? Could missing data, selection or your choice of test period change the conclusion?
- For which problems is your approach fine, and for which may it appear fine but not be useful?

## Ground rules

- You must use public data sources, and you must document where each one came from.
- Your project must address a real decision problem that arises in its setting.
- You must compare your ranking against a simple baseline (which may be one of your two approaches) using an observable evaluation, and you must not invent ground truth or treat unverified AI-generated judgements as observed outcomes.
- You must submit your dataset and a Python script that reproduces the principal results in your presentation, with brief instructions for running it and an identification of your data sources.
- You must compare two approaches on the same decision.

You may change your problem or approach as you learn what the data can support; no prior approval is needed, but explain your final choices and their limitations. There is no checkpoint or mid-course submission, and data sources are not claimed or reserved.

## Deliverables and dates

| Deliverable | Date |
|:------------|:-----|
| Team presentations: 12 minutes + 6 minutes of questions; order drawn by lot | Thu 26 and Fri 27 November |
| Final submission: dataset and Python script (with brief run instructions and data sources) | Tue 1 December, 11:59 pm |

The presentation should introduce the context and decision problem cleanly, then show your decision, the two shortlists and the evidence behind their comparison. Every team member presents and may be asked about any part of the analysis. The team receives one shared mark. The final submission is your dataset and Python script (with brief run instructions and data sources); no written report is required.

## How it will be graded

This is not about beating other teams; what matters is the clarity of your problem, how you tackled it, and what is still left untackled. The central question is whether your problem, data, method, evaluation and conclusions fit together. A complicated model is not evidence of a good solution; a simple method, carefully evaluated and understood, can earn marks.
