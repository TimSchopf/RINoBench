# Is this Idea Novel? An Automated Benchmark for Judgment of Research Ideas 🦏

> Repository of the LREC 2026 paper *"Is this Idea Novel? An Automated Benchmark for Judgment of Research Ideas"* 📄.

> It presents the **R**esearch **I**dea **No**velty Judgment **Bench**mark (*RINoBench* 🦏), a new evaluation benchmark including **a dataset of 1,381 research ideas derived from and judged by human experts as well as nine automated evaluation metrics** designed to assess both rubric-based novelty scores and textual justifications of novelty judgments.

📄 Paper: [tbp](tbp)

🤗 Data: [https://huggingface.co/datasets/TimSchopf/RINoBench](https://huggingface.co/datasets/TimSchopf/RINoBench)

## 🦏 Overview
Judging the novelty of research ideas is crucial for advancing science, enabling the identification of unexplored directions, and ensuring contributions meaningfully extend existing knowledge rather than reiterate minor variations. However, given the exponential growth of scientific literature, manually judging the novelty of research ideas through literature reviews is labor-intensive, subjective, and infeasible at scale. Therefore, recent efforts have proposed automated approaches for research idea novelty judgment. Yet, evaluation of these approaches remains largely inconsistent and is typically based on non-standardized human evaluations, hindering large-scale, comparable evaluations. 

To address this, we introduce **RINoBench 🦏**, the first comprehensive benchmark for large-scale evaluation of research idea novelty judgments.
Our benchmark unifies approaches for judging the novelty of research ideas by formalizing the task, illustrated in the Figure below, as the process of comparing a proposed idea with existing work to identify meaningful differences. Further, the task requires predicting a rubric-based novelty score (1–5) alongside a textual justification that grounds the judgment in related literature. This task design enables fine-grained, interpretable judgments of novelty and provides actionable feedback, empowering researchers to iteratively refine their ideas towards greater innovation and impact.

![example_novelty_judgment.png](figures/example_novelty_judgment.png)

This Figure illustrates the task setup of *RINoBench*. Given a research idea and its related works, a model must judge the novelty of the idea according to a five-point rubric. In addition, the model must provide a textual justification for its judgment, grounded in a comparison between the proposed research idea and the related works.


## 📂 Repository Structure

```
NoveltyBench/
├── data/
│   ├── final_benchmark_dataset                # includes the dataset of RINoBench
│   ├── evaluations                            # incudes the evaluation results of various state-of-the-art LLMs on RINoBench
│   └── novelty_predictions                    # incudes the reserach idea novelty judgments of various state-of-the-art LLMs on RINoBench
│
├── figures/                                   # includes the figures from the paper
│
├── src/
│   ├── data_processing                        # Scripts and LLM prompts used to construct our dataset
│   └── experiments                            # Scripts and LLM prompts used to generate LLM predictions as well as for evaluating the predictions
│
├── .gitignore
├── README.md 
└── requirements.txt  
```

## 💾 Data Description

| File                            | Description                     |
|---------------------------------|---------------------------------|
| `label_descriptions.json`       | The novelty judgment rubric.    |
| `train.json`                    | The train split of our dataset. |
| `test.json`                     | The test split of our dataset.  |

Each train and test split contains research ideas, gold novelty scores, gold textual judgment justifications, related works, and the respective sources from OpenReview.
