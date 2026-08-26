# AdaptiveHop

AdaptiveHop is a research-oriented NLP project on multi-hop question answering.
It investigates whether explicit question decomposition, iterative evidence
retrieval, and adaptive hop selection can improve answer accuracy, evidence
grounding, and retrieval efficiency over a conventional retrieve-then-answer
baseline.

## MuSiQue-Ans dataset foundation

The current implementation scope is dataset acquisition guidance, schema
validation, and exploratory analysis for MuSiQue-Ans. It intentionally does not
implement retrieval, answer generation, RAG, or model inference.

### Environment

Use Python 3.11 or later. Create a local environment and install the minimal
Task 1 dependencies:

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Dataset placement

Obtain MuSiQue-Ans from the [official MuSiQue repository](https://github.com/StonyBrookNLP/musique)
and place the supplied files at:

```text
data/musique_ans/train.jsonl
data/musique_ans/dev.jsonl
```

Dataset files are ignored by Git. See [the dataset README](data/musique_ans/README.md)
for expected split counts, provenance, and licensing guidance.

### Validation and EDA

After the dataset is in place, validate both splits with:

```powershell
python scripts/validate_musique.py
```

The EDA notebook is `notebooks/01_dataset_eda.ipynb`; launch it after validation:

```powershell
jupyter notebook notebooks/01_dataset_eda.ipynb
```

The loader, validator, tests, and notebook are added after the source data is
available so their schema assumptions can be checked against the actual release.
