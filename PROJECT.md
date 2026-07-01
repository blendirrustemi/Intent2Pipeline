# Agentic AI for Declarative Data Science Pipelines
## Lab 1 - Technical Specification

# Project Overview

This project is part of my Master's thesis.

The final goal is to develop an **Agentic AI system** capable of transforming natural language requests into deterministic declarative data science pipelines.

Unlike traditional LLM-powered systems that directly generate pipelines or code, this system should first understand the user's intent, estimate its confidence, ask clarification questions when necessary, and only then generate a deterministic pipeline specification.

The research contribution is reducing **pipeline drift** caused by different prompt wordings by introducing an interactive intent validation and canonicalization layer before pipeline generation.

This repository currently represents **Lab 1**, which is intended to build the complete project architecture and a partially functional prototype.

The implementation should be modular because this repository will continuously evolve into the final Master's thesis project.

---

# Final Vision (Thesis)

The final system should allow a user to:

1. Upload a dataset (CSV initially)
2. Describe a data science task in natural language
3. Detect ambiguous requests
4. Ask clarification questions
5. Build a canonical representation of the user's intent
6. Generate a deterministic pipeline
7. Execute the pipeline
8. Produce preprocessing outputs
9. Train baseline ML models
10. Evaluate the pipeline
11. Compare outputs across paraphrased prompts
12. Demonstrate reduced pipeline drift compared to existing approaches

The complete system should follow this architecture:

```
Dataset Upload
        │
        ▼
Dataset Analysis
        │
        ▼
User Prompt
        │
        ▼
Intent Understanding (LLM)
        │
        ▼
Confidence Evaluation
        │
 ┌──────┴────────┐
 │               │
High          Low
Confidence    Confidence
 │               │
 ▼               ▼
Continue     Ask Clarification
 │               │
 └──────User Reply─────────┐
                            │
                            ▼
                Canonical Intent Representation
                            │
                            ▼
            Deterministic Pipeline Generator
                            │
                            ▼
                Pipeline Execution Engine
                            │
                            ▼
                 Evaluation & Metrics
```

---

# Lab 1 Scope

Lab 1 **does NOT implement the complete thesis.**

The objective is to build a solid modular framework that demonstrates the proposed architecture.

Everything should be designed so additional features can be added later without restructuring the project.

---

# Technology Stack

Use:

- Python
- Streamlit
- Pandas
- Scikit-Learn
- OpenAI API (or provider abstraction)
- Plotly (optional)
- pathlib
- dataclasses when appropriate

The architecture should remain modular.

Avoid writing everything inside one file.

---

# Dataset

The initial dataset is the UCI Adult Income Dataset.

The dataset contains:

- numeric columns
- categorical columns
- missing values represented by '?'
- binary classification target (`income`)

This dataset will be used throughout the thesis.

---

# Required Project Structure

```
agentic-ai-pipeline/

│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│     └── adult.csv
│
├── outputs/
│     ├── canonical_intent.json
│     ├── pipeline_spec.json
│     ├── processed_dataset.csv
│     └── metrics.json
│
├── src/
│
│     ├── dataset/
│     │      loader.py
│     │      profiler.py
│     │
│     ├── intent/
│     │      extractor.py
│     │      confidence.py
│     │      clarification.py
│     │
│     ├── canonicalization/
│     │      canonicalizer.py
│     │
│     ├── pipeline/
│     │      generator.py
│     │      executor.py
│     │
│     ├── evaluation/
│     │      metrics.py
│     │
│     └── llm/
│            client.py
│
└── tests/
```

---

# Lab 1 Functional Requirements

## 1. Streamlit Interface

Create a simple chatbot-like interface.

The interface should include:

- CSV uploader
- Dataset overview
- Chat input
- Chat history
- Generated canonical intent
- Generated pipeline
- Execution results
- Metrics panel

The UI does NOT need to be beautiful.

Functionality is much more important.

---

## 2. Dataset Loader

After uploading a CSV:

Load the dataset.

Store it in memory.

Return:

- number of rows
- number of columns
- column names
- inferred column types
- missing value counts
- sample rows

---

## 3. Dataset Profiler

Automatically detect:

- numeric columns
- categorical columns
- missing values
- duplicated rows
- target candidates
- unique values

Return structured metadata.

Example:

```json
{
  "rows": 48842,
  "columns": 15,
  "numeric_columns": ["..."],
  "categorical_columns": ["..."],
  "missing_columns": ["..."],
  "possible_targets": ["..."]
}
```

---

## 4. User Prompt

The user should be able to type requests like:

> Prepare this dataset for classification.

or

> Clean this dataset and train a classifier.

The prompt is passed to the Intent module.

---

## 5. Intent Extraction Module

The LLM should extract structured intent.

Example:

```json
{
    "task_type": "classification",
    "target_column": null,
    "requires_cleaning": true,
    "requires_preprocessing": true,
    "requires_training": false
}
```

No pipeline generation should happen yet.

---

## 6. Confidence Evaluation

Evaluate whether enough information exists.

Example:

```json
{
    "confidence": 0.63,
    "missing_information": [
        "target_column"
    ]
}
```

---

## 7. Clarification Loop

If confidence is below a threshold:

Generate one clarification question.

Example:

> Which column should be predicted?

After the user answers:

Update the intent.

Recalculate confidence.

Repeat until sufficient confidence exists.

---

## 8. Canonical Intent

Convert the conversation into one stable representation.

Example:

```json
{
    "task_type":"classification",
    "target_column":"income",
    "missing_value_strategy":"median",
    "encoding":"one_hot",
    "split":"stratified"
}
```

This component is one of the main research contributions.

---

## 9. Pipeline Generator

Generate a deterministic pipeline specification.

Example:

```json
{
    "steps":[
        "replace_missing",
        "encode_categorical",
        "scale_numeric",
        "split_dataset"
    ]
}
```

Only generate structured pipeline specifications.

Do NOT generate arbitrary Python code.

---

## 10. Pipeline Execution

Execute the generated pipeline using deterministic Python functions.

For Lab 1, execution should support:

- replacing '?'
- missing value imputation
- categorical encoding
- scaling
- train/test split

Model training is optional for Lab 1.

---

## 11. Metrics Logger

Store information about every run.

Example:

```json
{
    "prompt":"",
    "confidence_before":0.61,
    "confidence_after":0.98,
    "questions_asked":1,
    "pipeline_generated":true,
    "execution_success":true,
    "latency":5.4
}
```

---

# Coding Principles

The project should follow:

- clean architecture
- modular design
- reusable components
- extensive comments
- type hints
- docstrings
- no duplicated code

Every module should be independently testable.

---

# Future Thesis Phases

These should NOT be implemented now, but the architecture should support them.

## Phase 2

Paraphrase stability evaluation.

Compare prompts such as:

- Prepare this dataset for classification.
- Build a classifier pipeline.
- Train a model to predict income.

Measure whether they produce identical canonical intents.

---

## Phase 3

Baseline comparison.

Compare:

Direct Prompt → Pipeline

vs

Prompt → Clarification → Canonical Intent → Pipeline

---

## Phase 4

Pipeline drift evaluation.

Metrics:

- Pipeline similarity
- Operator differences
- Parameter differences
- Execution consistency

---

## Phase 5

Model Training

Support:

- Logistic Regression
- Decision Tree
- Random Forest

Evaluate:

- Accuracy
- Precision
- Recall
- F1
- ROC AUC

---

## Phase 6

Support additional tasks.

Regression

Clustering

Time Series

Recommendation Systems

---

## Phase 7

Multi-agent architecture.

Possible agents:

- Dataset Analysis Agent
- Intent Agent
- Clarification Agent
- Pipeline Agent
- Evaluation Agent

---

# Goal for Lab 1

The final Lab 1 deliverable should demonstrate the complete architecture with a partially working implementation.

The emphasis is on showing how the components interact rather than implementing every possible feature.

The repository should already resemble the final thesis project so that future work mainly consists of improving each module instead of redesigning the architecture.