# causal-shapley

Implementation and notebooks accompanying the thesis *"Path-wise Causal Shapley Explanations of Machine-Learning Predictions"*. The repository contains:

- the Marginal / Conditional / Causal / Asymmetric Shapley library and the PE-SHAP / PW-SHAP path-wise implementations,
- synthetic SCM datasets for the canonical-structure comparison and for the path-wise benchmarks,
- the German Credit real-data case study,
- the Jupyter notebooks and headless scripts that produce the thesis figures and tables.

## Repository layout

```
causal-shapley/
├── README.md
├── requirements.txt                  pinned package versions
├── .gitignore
│
├── shapley_values/                   Library — Shapley implementations
│   ├── causal_shap.py                  Explainer + ShapleyValuesType
│   │                                   (MARGINAL / CONDITIONAL / CAUSAL)
│   ├── causal_shap_paper.py            CausalExplainer + EffectType
│   │                                   (paper-aligned causal + asymmetric causal)
│   ├── probabilities.py                Distribution / interventional helpers
│   ├── utils.py                        get_baseline() and shared helpers
│   └── exceptions.py
│
├── data/                             Datasets and SCM generators
│   ├── __init__.py                     Makes data/ a Python package
│   ├── simulations.py                  C1–C6 canonical structures
│   │                                   + S1, S2, S3 PE-SHAP benchmarks
│   └── german_credit.csv               Real-data case-study CSV
│
├── canonical_comparison/             Canonical-structure comparison notebooks
│   ├── marginal_shap.ipynb               Table tab:shapley-comparison
│   ├── conditional_shap.ipynb
│   ├── causal_shap.ipynb
│   ├── asymmetric_conditional_shap.ipynb
│   └── asymmetric_causal_shap.ipynb
│
├── path_wise_experiments/            PE-SHAP / PW-SHAP S1, S2, S3 benchmarks
│   ├── _common.py                      Shared estimator / plotting helpers
│   ├── experiment1_chain.py            Chain-mediation derivation (S1)
│   ├── experiment2_parallel.py         Parallel-mediator derivation (S2)
│   ├── run_main_table.py               Builds Table tab:main-results
│   ├── run_local_figure.py             Builds Figure fig:local-decomposition
│   └── path_wise/                      PW-SHAP estimation code
│
└── german_credit/                    Real-data case study
    ├── german_credit.ipynb               Interactive notebook
    └── german_credit_pe_pw.py             Standalone script (PE-SHAP + PW-SHAP)
```

## How experiments map to the thesis

| Section / figure / table | What it shows | Code location |
|---|---|---|
| Table `tab:shapley-comparison` | Marginal / Conditional / Asymmetric / Causal / Asymmetric Causal on C1–C6 | `canonical_comparison/*.ipynb` |
| Table `tab:main-results` | PE-SHAP vs PW-SHAP vs Causal Shapley on S1, S2, S3 | `path_wise_experiments/run_main_table.py` |
| Figure `fig:local-decomposition` | Per-sample PE-SHAP attribution on S2 | `path_wise_experiments/run_local_figure.py` |
| Appendix `sec:chain-mediation-proof` | Closed-form chain-mediation derivation | `path_wise_experiments/experiment1_chain.py` |
| Appendix `sec:parallel-mediation-proof` | Closed-form parallel-mediator derivation | `path_wise_experiments/experiment2_parallel.py` |
| German Credit case study | Counterfactual gender-bias attribution | `german_credit/german_credit.ipynb` |

## Setup

Python **3.12** is what the notebooks were tested on.

```bash
git clone <repo-url> causal-shapley
cd causal-shapley
python3.12 -m venv .venv
source .venv/bin/activate            # on Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Running the notebooks

Launch JupyterLab from the repository root so that `data/`, `shapley_values/`, and the experiment folders are all on `sys.path`:

```bash
cd causal-shapley
jupyter lab
```

Then open any notebook in `canonical_comparison/`, `german_credit/`, etc. Each notebook starts with a small `sys.path` bootstrap cell so it also works when the kernel is started inside the subdirectory.

## Running the headless scripts

All scripts must be run from the `causal-shapley/` root using the `-m` module syntax:

```bash
# Main table (S1, S2, S3)
python3.12 -m path_wise_experiments.run_main_table

# Local PE-SHAP decomposition figure
python3.12 -m path_wise_experiments.run_local_figure

# Closed-form chain mediation derivation
python3.12 -m path_wise_experiments.experiment1_chain

# Closed-form parallel-mediator derivation
python3.12 -m path_wise_experiments.experiment2_parallel

# German Credit case study (PE-SHAP + PW-SHAP)
python3.12 -m german_credit.german_credit_pe_pw
```

## Quick API reference

```python
# Marginal / Conditional Shapley
from shapley_values.causal_shap import Explainer, ShapleyValuesType

explainer = Explainer(X=X, model=model, is_classification=False,
                      rounding_precision=1, feature_names=["X1", "X2"])

phis = explainer.compute_shapley_values(sample, type=ShapleyValuesType.MARGINAL)
phis = explainer.compute_shapley_values(sample, type=ShapleyValuesType.CONDITIONAL)
phis = explainer.compute_shapley_values(sample, type=ShapleyValuesType.CONDITIONAL,
                                         is_asymmetric=True,
                                         causal_model={"X1": ["X2"]})

# Causal / Asymmetric Causal Shapley (paper-aligned API)
from shapley_values.causal_shap_paper import CausalExplainer, EffectType

explainer = CausalExplainer(X=X, model=model, is_classification=False,
                            rounding_precision=1, feature_names=["X1", "X2"])

phis = explainer.compute_shapley_values(
    sample,
    effect_type=EffectType.TOTAL,
    causal_model=[[0], [1]],         # tier list (feature indices)
    confounding=[False, False],       # one bool per tier
)

# Add `is_asymmetric=True` and an `asymmetric_causal_model` dict (integer-indexed)
# for the asymmetric variant:
phis = explainer.compute_shapley_values(
    sample,
    effect_type=EffectType.TOTAL,
    causal_model=[[0], [1]],
    confounding=[False, False],
    is_asymmetric=True,
    asymmetric_causal_model={0: [1]},
)
```

## Datasets

`data/simulations.py` exposes:

- **Canonical structures** for the C1–C6 comparison: `c1_chain`, `c2_fork`, `c3_unobserved_feature_confounder`, `c4_observed_feature_confounder`, `c5_collider`, `c6_unobserved_prediction_confounder`. Each returns `(features, y)` arrays with the SCM and sample point used in the thesis.
- **Path-wise benchmarks** S1, S2, S3: `chain_mediation_dataset`, `parallel_interaction_dataset`, `combined_realistic_dataset`, with their corresponding `*_ground_truth` helpers.

`data/german_credit.csv` is the German Credit dataset used by the case-study notebook.
