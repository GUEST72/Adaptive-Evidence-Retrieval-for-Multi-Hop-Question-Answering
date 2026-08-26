# MuSiQue-Ans Loader — Usage Guide

This document explains how to use the MuSiQue-Ans data loader provided in:

```text
src/data/musique_loader.py
```

The loader is designed to give the rest of the project a single, consistent way to load, validate, and access MuSiQue-Ans data.

The main idea is:

```text
JSONL files
    ↓
MuSiQue Loader
    ↓
Validated MuSiQueRecord objects
    ↓
Retrieval / Evaluation / Experiments
```

Other project components should use this loader instead of reading the JSONL files directly.

---

## 1. Dataset Location

By default, the loader expects the dataset in:

```text
data/
└── musique_ans/
    ├── train.jsonl
    └── dev.jsonl
```

The loader also supports the original MuSiQue release filenames:

```text
musique_ans_v1.0_train.jsonl
musique_ans_v1.0_dev.jsonl
```

# 2. Basic Usage

Import the loader:

```python
from src.data.musique_loader import load_split
```

Load the development set:

```python
dev_data = load_split("dev")
```

Load the training set:

```python
train_data = load_split("train")
```

The returned value is a list of `MuSiQueRecord` objects.

For example:

```python
print(len(dev_data))
```

Expected:

```text
2417
```

and:

```python
print(len(train_data))
```

Expected:

```text
19938
```

---

# 3. Working With a Record

Each item returned by `load_split()` is a `MuSiQueRecord`.

For example:

```python
record = dev_data[0]
```

You can access:

```python
record.id
record.question
record.answer
record.answer_aliases
record.paragraphs
record.question_decomposition
record.hop_count
```

Example:

```python
print(record.id)
print(record.question)
print(record.answer)
print(record.hop_count)
```

---

# 4. Hop Count

The loader defines hop count as:

```python
len(record.question_decomposition)
```

You can access it directly:

```python
record.hop_count
```

Possible values for MuSiQue-Ans are:

```text
2
3
4
```

For example:

```python
if record.hop_count == 4:
    print("This is a 4-hop question.")
```

This definition should be used consistently throughout the project.

---

# 5. Accessing Paragraphs

Each record contains a tuple of `Paragraph` objects.

You can access all paragraphs using:

```python
record.paragraphs
```

For example:

```python
for paragraph in record.paragraphs:
    print(paragraph.idx)
    print(paragraph.title)
    print(paragraph.paragraph_text)
    print(paragraph.is_supporting)
```

A paragraph contains:

```text
idx
title
paragraph_text
is_supporting
```

The `idx` value is the original paragraph index from the benchmark.

Do not replace this index with a new locally generated index. Later retrieval experiments depend on being able to map predictions back to the original benchmark paragraphs.

---

# 6. Getting Supporting Paragraphs

The loader provides a helper:

```python
from src.data.musique_loader import supporting_paragraphs
```

Usage:

```python
supporting = supporting_paragraphs(record)
```

This returns only the paragraphs where:

```python
paragraph.is_supporting is True
```

Example:

```python
for paragraph in supporting:
    print(paragraph.idx, paragraph.title)
```

This is useful for retrieval evaluation because the supporting paragraphs represent the benchmark's gold evidence.

---

# 7. Accessing the Question Decomposition

Each record contains its gold decomposition:

```python
record.question_decomposition
```

Each step contains:

```text
id
question
answer
paragraph_support_idx
```

Example:

```python
for step in record.question_decomposition:
    print("Step:", step.id)
    print("Question:", step.question)
    print("Answer:", step.answer)
    print("Supporting paragraph:", step.paragraph_support_idx)
```

This information is important for later multi-hop experiments.

For example, it allows us to understand:

```text
Original Question
       ↓
Hop 1
       ↓
Intermediate Answer
       ↓
Hop 2
       ↓
Final Answer
```

Do not modify the gold decomposition when using it for evaluation.

---

# 8. Getting One Question by ID

Instead of loading a split and manually searching for an ID, use:

```python
from src.data.musique_loader import get_question
```

Example:

```python
record = get_question("some_question_id")
```

By default, `get_question()` searches the development split.

You can explicitly select a split:

```python
record = get_question(
    "some_question_id",
    split="train"
)
```

If the question ID does not exist, the function raises:

```python
KeyError
```

---

# 9. Loading Without Automatic Validation

By default:

```python
load_split("dev")
```

performs dataset validation.

This is intentional.

If the dataset does not satisfy the expected schema or benchmark invariants, the loader raises:

```python
DatasetValidationError
```

You can disable validation when necessary:

```python
dev_data = load_split(
    "dev",
    validate=False
)
```

Use this option carefully.

It is mainly useful for:

* debugging;
* inspecting malformed records;
* exploratory work when you are explicitly investigating validation failures.

Normal experiments should use:

```python
validate=True
```

---

# 10. Using a Custom Data Directory

By default, the loader uses:

```text
data/musique_ans/
```

You can provide another directory:

```python
dev_data = load_split(
    "dev",
    data_dir="some/other/path"
)
```

You can also use an absolute path:

```python
dev_data = load_split(
    "dev",
    data_dir=r"C:\datasets\musique_ans"
)
```

The same option is available when using:

```python
get_question(...)
```

For example:

```python
record = get_question(
    "some_question_id",
    split="dev",
    data_dir=r"C:\datasets\musique_ans"
)
```

This is useful if the dataset is stored outside the GitHub repository.

---

# 11. Environment Variable for the Dataset Location

The loader also supports:

```text
MUSIQUE_DATA_DIR
```

You can set this environment variable when the dataset is stored somewhere else.

For example on Windows PowerShell:

```powershell
$env:MUSIQUE_DATA_DIR="C:\datasets\musique_ans"
```

On Linux / WSL:

```bash
export MUSIQUE_DATA_DIR="/home/user/datasets/musique_ans"
```

After setting it:

```python
dev_data = load_split("dev")
```

The loader will use the configured directory.

This is useful when:

* the dataset is not stored inside the repository;
* multiple team members use different local storage paths;
* large dataset files should remain outside Git.

---

# 12. Validation

The project also provides a command-line validation script:

```text
scripts/validate_musique.py
```

Run it from the repository root:

```powershell
python scripts/validate_musique.py
```

or:

```bash
python scripts/validate_musique.py
```

The validation checks important benchmark assumptions such as:

* required fields;
* valid question IDs;
* duplicate IDs;
* valid hop counts;
* valid paragraph structure;
* valid decomposition structure;
* valid supporting paragraph references;
* expected train/dev hop distributions.

A successful validation should report:

```text
Dataset validation completed successfully.
```

---

# 13. Expected Dataset Statistics

The validated MuSiQue-Ans release should contain:

### Train

```text
19,938 questions
```

with:

```text
2-hop: 14,376
3-hop:  4,387
4-hop:  1,175
```

### Dev

```text
2,417 questions
```

with:

```text
2-hop: 1,252
3-hop:   760
4-hop:   405
```

These numbers are useful as a quick sanity check after downloading or replacing the dataset.

---

# 14. Recommended Usage in Later Tasks

### Task 2 — BM25 Retrieval

Task 2 should load data through:

```python
from src.data.musique_loader import load_split

dev_data = load_split("dev")
```

Then use:

```python
record.paragraphs
```

to retrieve from the question's available context.

The retrieval implementation should preserve:

```python
paragraph.idx
```

so retrieval results can be compared directly with the benchmark's gold supporting paragraph indices.

---

### Task 3 — QA Baseline

Task 3 can use:

```python
dev_data = load_split("dev")
```

Then:

```python
for record in dev_data:
    question = record.question
    paragraphs = record.paragraphs
    gold_answer = record.answer
```

The baseline should not implement its own JSONL parser.

The loader is the shared interface between the dataset and the rest of the project.

---

# 15. Recommended Pattern

For most experiments, use:

```python
from src.data.musique_loader import load_split

data = load_split("dev")

for record in data:
    question = record.question

    # Use record.paragraphs for retrieval
    # Use record.answer for answer evaluation
    # Use record.question_decomposition for gold reasoning analysis
```

This keeps experiments simple and ensures that all project components interpret the dataset consistently.

---

# 16. What You Should NOT Do

Do not read the JSONL files directly in individual notebooks:

```python
with open("train.jsonl") as f:
    ...
```

Do not create a second custom parser for another task.

Do not manually reconstruct the dataset schema.

Do not modify `paragraph.idx`.

Do not modify gold `is_supporting` labels.

Do not alter the gold decomposition before evaluation.

Do not hardcode machine-specific absolute paths inside project code.

Use the shared loader instead.

---

# 17. Quick Reference

| Task                    | Function                             | Purpose                               |
| ----------------------- | ------------------------------------ | ------------------------------------- |
| Load train              | `load_split("train")`                | Load validated training data          |
| Load dev                | `load_split("dev")`                  | Load validated development data       |
| Disable validation      | `load_split("dev", validate=False)`  | Debug/inspect raw data                |
| Custom data path        | `load_split("dev", data_dir=...)`    | Load data from another location       |
| Get one question        | `get_question(id)`                   | Retrieve a specific benchmark example |
| Get supporting evidence | `supporting_paragraphs(record)`      | Get gold supporting paragraphs        |
| Hop count               | `record.hop_count`                   | Get 2/3/4-hop depth                   |
| Paragraphs              | `record.paragraphs`                  | Access all available context          |
| Decomposition           | `record.question_decomposition`      | Access gold hop structure             |
| Validate dataset        | `python scripts/validate_musique.py` | Check dataset integrity               |

---

# 18. Design Principle

The loader is intentionally kept as the **single entry point for MuSiQue data** within the project.

The rest of the system should operate on validated `MuSiQueRecord` objects instead of raw JSON.

This gives us one consistent interpretation of the benchmark and makes the later retrieval, QA, and evaluation components easier to maintain.

The loader is not responsible for retrieval or question answering.

Its responsibility ends at:

```text
Raw MuSiQue JSONL
        ↓
Load
        ↓
Validate
        ↓
Structured Records
        ↓
Ready for experiments
```

That separation should be preserved as the project grows.
