"""
Experiment 1 — Chain mediation.

DGP (no confounder):

    T  ~ Bernoulli(0.5)
    M1 = a1 * T + U1
    M2 = a2 * M1 + U2
    Y  = b1 * T + b2 * M2 + UY

Analytic chain-path ground truth  T -> M1 -> M2 -> Y  =  a1 * a2 * b2.

Usage:
    python -m experiments.experiment1_chain
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from simulations.simulations import (
    chain_mediation_dataset,
    chain_ground_truth,
    CHAIN_T_COL,
    CHAIN_M1_COL,
    CHAIN_M2_COL,
)
from experiments._common import train_model, split_train_eval
from path_wise.synthetic_attribution import (
    attribution_average,
    attribution_summary,
    local_decomposition,
    pick_median_control_sample,
)
from path_wise.causal_shapley_local import (
    causal_shapley_row,
    causal_shapley_direct_indirect_row,
)


def run_once(a1=0.8, a2=0.7, b1=0.5, b2=1.0, sigma=0.5,
             n_total=6000, n_eval=1000, seed=0, model_type="mlp"):
    X, y = chain_mediation_dataset(
        n=n_total, a1=a1, a2=a2, b1=b1, b2=b2, sigma=sigma, seed=seed,
    )
    X_train, y_train, X_eval, y_eval = split_train_eval(
        X, y, eval_size=n_eval, seed=seed,
    )
    model = train_model(X_train, y_train, model_type=model_type)

    # Evaluate path attributions averaged over control-arm held-out samples.
    control_mask = X_eval[:, CHAIN_T_COL] == 0
    X_eval_control = X_eval[control_mask]

    per_sample = attribution_average(
        model, X_train, T_col=CHAIN_T_COL,
        mediator_cols=[CHAIN_M1_COL, CHAIN_M2_COL],
        X_eval=X_eval_control,
        base_cond_cols=(),
        eps_mediator=0.2,
    )
    summary = attribution_summary(per_sample)

    gt = chain_ground_truth(a1=a1, a2=a2, b1=b1, b2=b2)
    return summary, gt


def local_chain_bars(a1=0.8, a2=0.7, b1=0.5, b2=1.0, sigma=0.5,
                     n_total=6000, n_eval=1000, seed=0,
                     model_type="mlp", eps_mediator=0.2):
    """
    Train the chain model and return the bar-ready local PE-SHAP
    decomposition at the median control-arm sample.

    Output dict keys:

        sample_idx       index of the chosen row in X_eval
        sample           the raw feature vector (T, M1, M2)
        v_free           v_block(emptyset)(x)       local total effect
        v_all            v_block(all)(x)            local direct effect (CDE)
        lam_m1 / lam_m2  raw path-wise PE values    (overshoot when interacting)
        phi_m1 / phi_m2  Shapley-decomposed PE      (sum to local_mediated)
        local_mediated   v_free - v_all             local total indirect effect
        interaction_surplus  raw_sum - local_mediated  (expected 0 for the chain SEM)
        gt_chain         analytic a1 * a2 * b2      reference for captioning
    """
    X, y = chain_mediation_dataset(
        n=n_total, a1=a1, a2=a2, b1=b1, b2=b2, sigma=sigma, seed=seed,
    )
    X_train, y_train, X_eval, _y_eval = split_train_eval(
        X, y, eval_size=n_eval, seed=seed,
    )
    model = train_model(X_train, y_train, model_type=model_type)

    idx, sample = pick_median_control_sample(
        X_eval, T_col=CHAIN_T_COL,
        mediator_cols=[CHAIN_M1_COL, CHAIN_M2_COL],
    )

    dec = local_decomposition(
        model, X_train, T_col=CHAIN_T_COL,
        mediator_cols=[CHAIN_M1_COL, CHAIN_M2_COL],
        sample=sample,
        base_cond_cols=(),
        eps_mediator=eps_mediator,
    )

    gt = chain_ground_truth(a1=a1, a2=a2, b1=b1, b2=b2)
    return {
        "sample_idx": idx,
        "sample": dec["sample"],
        "v_free": dec["v_free"],
        "v_all": dec["v_all"],
        "lam_m1": dec["lam"][CHAIN_M1_COL],
        "lam_m2": dec["lam"][CHAIN_M2_COL],
        "phi_m1": dec["phi"][CHAIN_M1_COL],
        "phi_m2": dec["phi"][CHAIN_M2_COL],
        "local_mediated": dec["local_mediated"],
        "interaction_surplus": dec["interaction_surplus"],
        "gt_chain": gt["chain_t_m1_m2_y"],
        "gt_direct": gt["direct"],
        "gt_total": gt["total"],
    }


def local_chain_causal_shap(a1=0.8, a2=0.7, b1=0.5, b2=1.0, sigma=0.5,
                             n_total=6000, n_eval=1000, seed=0,
                             model_type="mlp", n_mc=400, eps=0.25):
    """
    Train the chain model and compute per-feature symmetric Causal
    Shapley values (Heskes 2020) at the median control-arm sample, under
    the chain DAG  T -> M1 -> M2.

    Returns a dict with the per-feature values, the model baseline, the
    local prediction f(x), and the efficiency gap
    ``phi.sum() - (f(x) - baseline)`` for diagnostics.
    """
    X, y = chain_mediation_dataset(
        n=n_total, a1=a1, a2=a2, b1=b1, b2=b2, sigma=sigma, seed=seed,
    )
    X_train, y_train, X_eval, _y_eval = split_train_eval(
        X, y, eval_size=n_eval, seed=seed,
    )
    model = train_model(X_train, y_train, model_type=model_type)

    idx, sample = pick_median_control_sample(
        X_eval, T_col=CHAIN_T_COL,
        mediator_cols=[CHAIN_M1_COL, CHAIN_M2_COL],
    )

    dag_parents = {
        CHAIN_T_COL: [],
        CHAIN_M1_COL: [CHAIN_T_COL],
        CHAIN_M2_COL: [CHAIN_M1_COL],
    }
    rng = np.random.default_rng(seed)
    phi, v_table = causal_shapley_row(
        model, X_train, sample, dag_parents,
        n_mc=n_mc, eps=eps, rng=rng,
    )

    baseline = v_table[frozenset()]
    f_x = float(np.asarray(model.predict(sample.reshape(1, -1))).ravel()[0])
    return {
        "sample_idx": idx,
        "sample": sample,
        "phi_T": float(phi[CHAIN_T_COL]),
        "phi_M1": float(phi[CHAIN_M1_COL]),
        "phi_M2": float(phi[CHAIN_M2_COL]),
        "phi_sum": float(phi.sum()),
        "baseline": baseline,
        "f_x": f_x,
        "efficiency_gap": float(phi.sum() - (f_x - baseline)),
        "v_table": {tuple(sorted(s)): v for s, v in v_table.items()},
    }


def local_chain_causal_shap_direct_indirect(
    a1=0.8, a2=0.7, b1=0.5, b2=1.0, sigma=0.5,
    n_total=6000, n_eval=1000, seed=0,
    model_type="mlp", n_mc=400, eps=0.25,
):
    """
    Train the chain model and compute the Heskes direct / indirect
    decomposition of per-feature symmetric Causal Shapley values at the
    median control-arm sample, under the chain DAG  T -> M1 -> M2.

    Expected structural pattern in the chain SEM:

        M1: phi_direct = 0              (no direct edge M1 -> Y)
        M2: phi_indirect = 0            (M2 has no descendants)
        T : split between direct (b1*T) and indirect (chain path).
    """
    X, y = chain_mediation_dataset(
        n=n_total, a1=a1, a2=a2, b1=b1, b2=b2, sigma=sigma, seed=seed,
    )
    X_train, y_train, X_eval, _y_eval = split_train_eval(
        X, y, eval_size=n_eval, seed=seed,
    )
    model = train_model(X_train, y_train, model_type=model_type)

    idx, sample = pick_median_control_sample(
        X_eval, T_col=CHAIN_T_COL,
        mediator_cols=[CHAIN_M1_COL, CHAIN_M2_COL],
    )

    dag_parents = {
        CHAIN_T_COL: [],
        CHAIN_M1_COL: [CHAIN_T_COL],
        CHAIN_M2_COL: [CHAIN_M1_COL],
    }
    rng = np.random.default_rng(seed)
    phi_total, phi_direct, phi_indirect, v_table = (
        causal_shapley_direct_indirect_row(
            model, X_train, sample, dag_parents,
            n_mc=n_mc, eps=eps, rng=rng,
        )
    )

    baseline = v_table[frozenset()]
    f_x = float(np.asarray(model.predict(sample.reshape(1, -1))).ravel()[0])
    return {
        "sample_idx": idx,
        "sample": sample,
        "phi_T_total": float(phi_total[CHAIN_T_COL]),
        "phi_M1_total": float(phi_total[CHAIN_M1_COL]),
        "phi_M2_total": float(phi_total[CHAIN_M2_COL]),
        "phi_T_direct": float(phi_direct[CHAIN_T_COL]),
        "phi_M1_direct": float(phi_direct[CHAIN_M1_COL]),
        "phi_M2_direct": float(phi_direct[CHAIN_M2_COL]),
        "phi_T_indirect": float(phi_indirect[CHAIN_T_COL]),
        "phi_M1_indirect": float(phi_indirect[CHAIN_M1_COL]),
        "phi_M2_indirect": float(phi_indirect[CHAIN_M2_COL]),
        "phi_total_sum": float(phi_total.sum()),
        "baseline": baseline,
        "f_x": f_x,
        "efficiency_gap": float(phi_total.sum() - (f_x - baseline)),
    }


def run_many(n_runs=20, base_seed=0, **kwargs):
    """
    Outer Monte Carlo loop around ``run_once``.  Each iteration draws a
    fresh dataset, retrains the model, and recomputes the attribution
    averages.  Returns a dict keyed by attribution name with arrays of
    length ``n_runs`` (one per-run mean per element), plus the final
    ground-truth dict.
    """
    per_run = None
    gt = None
    for k in range(n_runs):
        summary, gt = run_once(seed=base_seed + k, **kwargs)
        if per_run is None:
            per_run = {key: [] for key in summary.keys()}
        for key, stats in summary.items():
            per_run[key].append(stats["mean"])
    per_run = {k: np.asarray(v, dtype=float) for k, v in per_run.items()}
    return per_run, gt


def mc_summary(per_run):
    """Mean and Monte Carlo std (across independent runs) per attribution key."""
    out = {}
    for k, arr in per_run.items():
        out[k] = {
            "mean": float(np.nanmean(arr)),
            "mc_std": float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "n_runs": int(len(arr)),
        }
    return out


def chain_table_mc(mc, gt):
    """Headline table using Monte Carlo error bars across independent runs."""
    rows = [{
        "Method": "PE-SHAP",
        "T -> M1 -> M2 -> Y (via block M1)": mc[f"pe_m{CHAIN_M1_COL}"]["mean"],
        "mc_std": mc[f"pe_m{CHAIN_M1_COL}"]["mc_std"],
        "via block M2": mc[f"pe_m{CHAIN_M2_COL}"]["mean"],
        "GT": gt["chain_t_m1_m2_y"],
    }, {
        "Method": "PW-SHAP",
        "T -> M1 -> M2 -> Y (via block M1)": mc[f"pw_m{CHAIN_M1_COL}"]["mean"],
        "mc_std": mc[f"pw_m{CHAIN_M1_COL}"]["mc_std"],
        "via block M2": mc[f"pw_m{CHAIN_M2_COL}"]["mean"],
        "GT": gt["chain_t_m1_m2_y"],
    }, {
        "Method": "Causal indirect (lumped)",
        "T -> M1 -> M2 -> Y (via block M1)": mc["causal_indirect"]["mean"],
        "mc_std": mc["causal_indirect"]["mc_std"],
        "via block M2": mc["causal_indirect"]["mean"],
        "GT": gt["chain_t_m1_m2_y"],
    }]
    return pd.DataFrame(rows)


def chain_table(summary, gt):
    """Assemble the headline attribution table for the chain path."""
    rows = []
    # In the chain DGP, PE(M1) == PE(M2) is the chain-path effect, and
    # PW(M1) == PW(M2) == 0.  We report both as a consistency check.
    rows.append({
        "Method": "PE-SHAP",
        "T -> M1 -> M2 -> Y (via block M1)": summary[f"pe_m{CHAIN_M1_COL}"]["mean"],
        "std": summary[f"pe_m{CHAIN_M1_COL}"]["std"],
        "via block M2": summary[f"pe_m{CHAIN_M2_COL}"]["mean"],
        "GT": gt["chain_t_m1_m2_y"],
    })
    rows.append({
        "Method": "PW-SHAP",
        "T -> M1 -> M2 -> Y (via block M1)": summary[f"pw_m{CHAIN_M1_COL}"]["mean"],
        "std": summary[f"pw_m{CHAIN_M1_COL}"]["std"],
        "via block M2": summary[f"pw_m{CHAIN_M2_COL}"]["mean"],
        "GT": gt["chain_t_m1_m2_y"],
    })
    rows.append({
        "Method": "Causal indirect (lumped)",
        "T -> M1 -> M2 -> Y (via block M1)": summary["causal_indirect"]["mean"],
        "std": summary["causal_indirect"]["std"],
        "via block M2": summary["causal_indirect"]["mean"],
        "GT": gt["chain_t_m1_m2_y"],
    })
    return pd.DataFrame(rows)


def a2_sweep(a2_grid=(0.0, 0.2, 0.5, 1.0, 2.0), **kwargs):
    """Sweep the second-link coefficient; one curve per method."""
    records = []
    for a2 in a2_grid:
        summary, gt = run_once(a2=a2, **kwargs)
        records.append({
            "a2": a2,
            "PE-SHAP": summary[f"pe_m{CHAIN_M1_COL}"]["mean"],
            "PW-SHAP": summary[f"pw_m{CHAIN_M1_COL}"]["mean"],
            "Causal indirect": summary["causal_indirect"]["mean"],
            "GT": gt["chain_t_m1_m2_y"],
        })
    return pd.DataFrame(records)


def plot_a2_sweep(df, path=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["a2"], df["GT"], "--", color="black", label="Analytic GT")
    ax.plot(df["a2"], df["PE-SHAP"], "o-", label="PE-SHAP")
    ax.plot(df["a2"], df["PW-SHAP"], "s-", label="PW-SHAP")
    ax.plot(df["a2"], df["Causal indirect"], "^-", label="Causal indirect")
    ax.set_xlabel(r"$a_2$ (M1 -> M2 link strength)")
    ax.set_ylabel(r"Attribution for $T\to M_1\to M_2\to \hat Y$")
    ax.set_title("Experiment 1: chain-path attribution vs. $a_2$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig


if __name__ == "__main__":
    print("=== Experiment 1: Chain mediation (single-run spread) ===")
    summary, gt = run_once()
    df_table = chain_table(summary, gt)
    print(df_table.to_string(index=False))

    print("\n=== Experiment 1: Monte Carlo across 20 independent runs ===")
    per_run, gt_mc = run_many(n_runs=20)
    mc = mc_summary(per_run)
    print(chain_table_mc(mc, gt_mc).to_string(index=False))

    print("\n=== Experiment 1: local bar decomposition at median control sample ===")
    bars = local_chain_bars()
    for k, v in bars.items():
        if isinstance(v, (int, float)):
            print(f"  {k:22s} = {v:+.4f}")
        else:
            print(f"  {k:22s} = {v}")

    print("\n=== Experiment 1: Heskes Causal Shapley at median control sample ===")
    cs = local_chain_causal_shap()
    for k, v in cs.items():
        if k == "v_table":
            print(f"  {k}:")
            for s, vs in v.items():
                print(f"    v({list(s)}) = {vs:+.4f}")
        elif isinstance(v, (int, float)):
            print(f"  {k:22s} = {v:+.4f}")
        else:
            print(f"  {k:22s} = {v}")

    print("\n=== Experiment 1: Heskes direct / indirect decomposition ===")
    cs_di = local_chain_causal_shap_direct_indirect()
    for k, v in cs_di.items():
        if isinstance(v, (int, float)):
            print(f"  {k:22s} = {v:+.4f}")
        else:
            print(f"  {k:22s} = {v}")

    print("\n=== Experiment 1: a2 sweep ===")
    df_sweep = a2_sweep()
    print(df_sweep.to_string(index=False))

    fig = plot_a2_sweep(df_sweep, path="experiments/experiment1_a2_sweep.png")
    print("Saved figure to experiments/experiment1_a2_sweep.png")
