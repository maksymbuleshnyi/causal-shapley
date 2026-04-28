# causal-shapley

Code for the thesis *Path-wise Causal Shapley Explanations of Machine-Learning Predictions*.

## Layout

```
shapley_values/         core Shapley library (Marginal / Conditional / Causal / Asymmetric)
data/                   SCM generators (C1-C6, S1, S2, S3) and the German Credit CSV
canonical_comparison/   notebooks: Shapley-variant comparison on C1-C6
path_wise_experiments/  scripts: PE-SHAP / PW-SHAP / Causal Shapley on S1, S2, S3
pw-causal-comparison/   script: PW vs Causal Shapley across mediation structures
german_credit/          case study: Sex attribution
```

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run everything

```bash
python run_all_experiments.py
```

Each experiment writes its summary to `./results/<name>.txt` next to the script.

## Run individually

```bash
cd pw-causal-comparison    && python mediation_structures_comparison.py
cd german_credit           && python german_credit_attribution.py
cd path_wise_experiments   && python compare_methods_on_benchmarks.py
cd path_wise_experiments   && python local_attributions_s2.py
```

## Notebooks

```bash
jupyter lab
```

Open anything in `canonical_comparison/` or `german_credit/`.
