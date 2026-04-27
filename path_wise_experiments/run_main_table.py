"""
Main-table runner for the path-wise attribution experiments.

Runs the three benchmarks end to end and emits a LaTeX-ready fragment:

  S1  chain mediation              (Experiment 1 — isolates PW-SHAP's
                                    structural zero on a chain path)
  S2  parallel mediators with
      outcome interaction          (Experiment 2 — isolates the
                                    interaction-driven magnitude / sign
                                    error on the T -> M1 path)
  S3  combined realistic DAG       (Experiment 3 — confounding,
                                    nonlinearity and both mediation
                                    patterns simultaneously; the
                                    failure modes compose)

For each benchmark we do R independent repetitions.  Every repetition
draws a fresh synthetic dataset from the SEM, retrains the black-box
model (MLP by default), and averages per-sample attributions over the
held-out control-arm samples.  We then report the grand mean across
the R repetitions and the Monte Carlo standard deviation across them
— the reproducibility of the reported number under retraining and
resampling.

Usage:
    python -m experiments.run_main_table

Defaults are tuned for a quick smoke run (``n_runs=5``); bump
``n_runs`` to 20 for the thesis-final numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data.simulations import (
    chain_mediation_dataset,
    CHAIN_T_COL, CHAIN_M1_COL, CHAIN_M2_COL,
    parallel_interaction_dataset,
    PAR_T_COL, PAR_M1_COL, PAR_M2_COL,
    combined_realistic_dataset,
    COMB_T_COL, COMB_C_COL, COMB_M1_COL, COMB_M2_COL,
)
from path_wise_experiments._common import train_model, split_train_eval
from path_wise_experiments.path_wise.synthetic_attribution import (
    attribution_average, attribution_summary,
)
from path_wise_experiments.path_wise.causal_shapley_local import causal_shapley_row


# =============================================================================
# Path decomposition on a 2-mediator DAG
# =============================================================================
#
# Both PE-SHAP and PW-SHAP use single-mediator formulas against the
# ATE baseline v_free = v_block(emptyset):
#
#     PE-SHAP(T -> M_k -> Y) = v_free - v_block({M_k})
#     PW-SHAP(T -> M_k -> Y) = v_all  - v_block(all \ {M_k})
#     direct(T -> Y)         = v_all
#
# In a DAG with a chain sub-path (e.g. T -> M1 -> M2 -> Y in S3),
# pinning M_k severs every T-path that routes through M_k at once, so
# PE-SHAP(T -> M_1 -> Y) lumps the direct M1 edge together with any
# chain segment that passes through M1.  We therefore do not attempt
# to report a separate chain column.
# =============================================================================


# =============================================================================
# NIE on the trained model (method-independent ground truth)
# =============================================================================
#
# For each control-arm sample we replay the SEM under T=0 and T=1,
# keeping the same noise draws, then feed the counterfactual rows
# through the trained model.  The NIE through a mediator M_k is
#
#     NIE_Yhat(M_k) = E[ Yhat(T=1, M_k(1), M_{-k}(0))
#                      - Yhat(T=1, M_k(0), M_{-k}(0)) ]
#
# This is the natural indirect effect of the *model*, not the true Y.
# It uses no attribution method — only counterfactual predictions.


def _nie_s1(model, X_train, s1_params, n_samples=2000, seed=42):
    """NIE_Yhat for the chain path T -> M1 -> M2 -> Y."""
    rng = np.random.default_rng(seed)
    a1, a2 = s1_params["a1"], s1_params["a2"]
    sigma = s1_params["sigma"]
    U1 = rng.normal(0, sigma, n_samples)
    U2 = rng.normal(0, sigma, n_samples)
    M1_0 = a1 * 0 + U1
    M1_1 = a1 * 1 + U1
    M2_under_M1_0 = a2 * M1_0 + U2
    M2_under_M1_1 = a2 * M1_1 + U2
    T_ones = np.ones(n_samples)
    row_base = np.column_stack([T_ones, M1_0, M2_under_M1_0])
    row_do   = np.column_stack([T_ones, M1_1, M2_under_M1_1])
    return float(np.mean(model.predict(row_do) - model.predict(row_base)))


def _nie_s2(model, X_train, s2_params, n_samples=2000, seed=42):
    """Total NIE_Yhat for the parallel DGP (shift both M1 and M2)."""
    rng = np.random.default_rng(seed)
    alpha1, alpha2 = s2_params["alpha1"], s2_params["alpha2"]
    sigma = s2_params["sigma"]
    U1 = rng.normal(0, sigma, n_samples)
    U2 = rng.normal(0, sigma, n_samples)
    M1_0 = alpha1 * 0 + U1
    M1_1 = alpha1 * 1 + U1
    M2_0 = alpha2 * 0 + U2
    M2_1 = alpha2 * 1 + U2
    T_ones = np.ones(n_samples)
    row_base = np.column_stack([T_ones, M1_0, M2_0])
    row_do   = np.column_stack([T_ones, M1_1, M2_1])
    return float(np.mean(model.predict(row_do) - model.predict(row_base)))


def _nie_s3(model, X_train, n_samples=2000, seed=42):
    """Total NIE_Yhat for the combined DAG (all mediators shift together)."""
    rng = np.random.default_rng(seed)
    sigma = 0.5
    C = rng.normal(0.3, 0.5, n_samples)
    U1 = rng.normal(0, sigma, n_samples)
    U2 = rng.normal(0, sigma, n_samples)
    M1_0 = 0.5 * 0 + U1
    M1_1 = 0.5 * 1 + U1
    M2_under_M1_0_T0 = 0.7 * M1_0 + 0.3 * 0 + U2
    M2_under_M1_1_T1 = 0.7 * M1_1 + 0.3 * 1 + U2
    T_ones = np.ones(n_samples)
    T_zeros = np.zeros(n_samples)
    row_t1_all0 = np.column_stack([T_ones, C, M1_0, M2_under_M1_0_T0])
    row_t0_all0 = np.column_stack([T_zeros, C, M1_0, M2_under_M1_0_T0])
    row_t1_all1 = np.column_stack([T_ones, C, M1_1, M2_under_M1_1_T1])
    direct = float(np.mean(model.predict(row_t1_all0) - model.predict(row_t0_all0)))
    total = float(np.mean(model.predict(row_t1_all1) - model.predict(row_t0_all0)))
    return float(total - direct)


def _pe_paths(summary, m1_col, m2_col):
    return {
        "direct": summary["direct_cde"]["mean"],
        "path_m1": summary[f"pe_m{m1_col}"]["mean"],
        "path_m2": summary[f"pe_m{m2_col}"]["mean"],
    }


def _pw_paths(summary, m1_col, m2_col):
    return {
        "direct": summary["direct_cde"]["mean"],
        "path_m1": summary[f"pw_m{m1_col}"]["mean"],
        "path_m2": summary[f"pw_m{m2_col}"]["mean"],
    }


# =============================================================================
# Per-benchmark single-run estimators
# =============================================================================


def _run_s1(seed, a1, a2, b1, b2, sigma, n_total, n_eval, model_type):
    X, y = chain_mediation_dataset(
        n=n_total, a1=a1, a2=a2, b1=b1, b2=b2, sigma=sigma, seed=seed,
    )
    X_train, y_train, X_eval, _ = split_train_eval(
        X, y, eval_size=n_eval, seed=seed,
    )
    model = train_model(X_train, y_train, model_type=model_type)
    ctrl = X_eval[X_eval[:, CHAIN_T_COL] == 0]
    per = attribution_average(
        model, X_train, CHAIN_T_COL, [CHAIN_M1_COL, CHAIN_M2_COL],
        ctrl, base_cond_cols=(),
    )
    s = attribution_summary(per)
    nie = _nie_s1(model, X_train, dict(a1=a1, a2=a2, b1=b1, b2=b2, sigma=sigma))
    return {
        "pe": s[f"pe_m{CHAIN_M1_COL}"]["mean"],
        "pw": s[f"pw_m{CHAIN_M1_COL}"]["mean"],
        "cs": s["causal_indirect"]["mean"],
        "nie": nie,
    }


def _run_s2(seed, alpha1, alpha2, beta1, beta2, beta3, beta4, beta5,
            sigma, n_total, n_eval, model_type,
            cs_n_eval, cs_n_mc, cs_eps):
    X, y = parallel_interaction_dataset(
        n=n_total, alpha1=alpha1, alpha2=alpha2,
        beta1=beta1, beta2=beta2, beta3=beta3, beta4=beta4, beta5=beta5,
        sigma=sigma, seed=seed,
    )
    X_train, y_train, X_eval, _ = split_train_eval(
        X, y, eval_size=n_eval, seed=seed,
    )
    model = train_model(X_train, y_train, model_type=model_type)
    ctrl = X_eval[X_eval[:, PAR_T_COL] == 0]
    per = attribution_average(
        model, X_train, PAR_T_COL, [PAR_M1_COL, PAR_M2_COL],
        ctrl, base_cond_cols=(),
    )
    s = attribution_summary(per)
    pe = _pe_paths(s, PAR_M1_COL, PAR_M2_COL)
    pw = _pw_paths(s, PAR_M1_COL, PAR_M2_COL)
    # Causal SHAP: per-feature symmetric Heskes value, averaged over
    # a subset of control-arm samples.
    dag_parents = {
        PAR_T_COL: [],
        PAR_M1_COL: [PAR_T_COL],
        PAR_M2_COL: [PAR_T_COL],
    }
    rng = np.random.default_rng(seed + 10_000)
    n_cs = int(min(len(ctrl), cs_n_eval))
    phi_m1_vals = []
    phi_m2_vals = []
    for i in range(n_cs):
        phi, _ = causal_shapley_row(
            model, X_train, ctrl[i], dag_parents,
            n_mc=cs_n_mc, eps=cs_eps, rng=rng,
        )
        phi_m1_vals.append(float(phi[PAR_M1_COL]))
        phi_m2_vals.append(float(phi[PAR_M2_COL]))
    nie = _nie_s2(model, X_train, dict(
        alpha1=alpha1, alpha2=alpha2, sigma=sigma,
    ))
    return {
        "pe_direct": pe["direct"],
        "pe_path_m1": pe["path_m1"],
        "pe_path_m2": pe["path_m2"],
        "pe_indirect": s["causal_indirect"]["mean"],
        "pw_direct": pw["direct"],
        "pw_path_m1": pw["path_m1"],
        "pw_path_m2": pw["path_m2"],
        "pw_indirect": -s["causal_indirect"]["mean"],
        "cs_indirect": float(np.mean(phi_m1_vals)) + float(np.mean(phi_m2_vals)),
        "nie": nie,
    }


def _run_s3(seed, sigma, n_total, n_eval, model_type):
    X, y = combined_realistic_dataset(n=n_total, sigma=sigma, seed=seed)
    X_train, y_train, X_eval, _ = split_train_eval(
        X, y, eval_size=n_eval, seed=seed,
    )
    model = train_model(X_train, y_train, model_type=model_type)
    ctrl = X_eval[X_eval[:, COMB_T_COL] == 0]
    per = attribution_average(
        model, X_train, COMB_T_COL, [COMB_M1_COL, COMB_M2_COL],
        ctrl, base_cond_cols=(COMB_C_COL,),
    )
    s = attribution_summary(per)
    pe = _pe_paths(s, COMB_M1_COL, COMB_M2_COL)
    pw = _pw_paths(s, COMB_M1_COL, COMB_M2_COL)
    nie = _nie_s3(model, X_train)
    return {
        "pe_direct": pe["direct"],
        "pe_path_m1": pe["path_m1"],
        "pe_path_m2": pe["path_m2"],
        "pe_indirect": s["causal_indirect"]["mean"],
        "pw_direct": pw["direct"],
        "pw_path_m1": pw["path_m1"],
        "pw_path_m2": pw["path_m2"],
        "pw_indirect": -s["causal_indirect"]["mean"],
        "cs_direct": s["direct_cde"]["mean"],
        "cs_lumped_indirect": s["causal_indirect"]["mean"],
        "nie": nie,
    }


# =============================================================================
# Monte Carlo aggregation
# =============================================================================


def _mc_aggregate(records):
    out = {}
    for k, vs in records.items():
        arr = np.asarray(vs, dtype=float)
        mean = float(np.nanmean(arr))
        std = float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else 0.0
        out[k] = (mean, std)
    return out


@dataclass
class RunConfig:
    n_runs: int = 20
    n_total: int = 10000
    n_eval: int = 500
    model_type: str = "mlp"
    # S2 Causal SHAP: inner-loop size.  Kept small because each call
    # involves a full 2^n_features coalition sweep with Monte Carlo
    # draws per coalition.
    s2_cs_n_eval: int = 30
    s2_cs_n_mc: int = 200
    s2_cs_eps: float = 0.25


def run_all(cfg: RunConfig | None = None, s1_params=None, s2_params=None):
    cfg = cfg or RunConfig()
    s1_params = s1_params or dict(
        a1=0.8, a2=0.7, b1=0.5, b2=1.0, sigma=0.5,
    )
    s2_params = s2_params or dict(
        alpha1=0.5, alpha2=0.7,
        beta1=0.3, beta2=0.2, beta3=-1.1, beta4=0.2, beta5=0.8,
        sigma=0.5,
    )

    # ----------------------------------------------------------------- S1
    print(f"[S1] chain mediation  ({cfg.n_runs} repetitions)")
    s1_recs = {"pe": [], "pw": [], "cs": [], "nie": []}
    for r in range(cfg.n_runs):
        out = _run_s1(
            seed=r, n_total=cfg.n_total, n_eval=cfg.n_eval,
            model_type=cfg.model_type, **s1_params,
        )
        for k, v in out.items():
            s1_recs[k].append(v)
        print(f"  run {r + 1}/{cfg.n_runs}  pe={out['pe']:+.3f}  "
              f"pw={out['pw']:+.3f}  cs={out['cs']:+.3f}  "
              f"nie={out['nie']:+.3f}")
    s1 = _mc_aggregate(s1_recs)

    # ----------------------------------------------------------------- S2
    print(f"\n[S2] parallel with interaction  ({cfg.n_runs} repetitions)")
    s2_keys = [
        "pe_direct", "pe_path_m1", "pe_path_m2", "pe_indirect",
        "pw_direct", "pw_path_m1", "pw_path_m2", "pw_indirect",
        "cs_indirect", "nie",
    ]
    s2_recs = {k: [] for k in s2_keys}
    for r in range(cfg.n_runs):
        out = _run_s2(
            seed=r, n_total=cfg.n_total, n_eval=cfg.n_eval,
            model_type=cfg.model_type,
            cs_n_eval=cfg.s2_cs_n_eval,
            cs_n_mc=cfg.s2_cs_n_mc,
            cs_eps=cfg.s2_cs_eps,
            **s2_params,
        )
        for k in s2_keys:
            s2_recs[k].append(out[k])
        print(f"  run {r + 1}/{cfg.n_runs}  "
              f"pe[m1,m2]=({out['pe_path_m1']:+.3f},{out['pe_path_m2']:+.3f})  "
              f"nie={out['nie']:+.3f}")
    s2 = _mc_aggregate(s2_recs)

    # ----------------------------------------------------------------- S3
    print(f"\n[S3] combined realistic DAG  ({cfg.n_runs} repetitions)")
    s3_keys = [
        "pe_direct", "pe_path_m1", "pe_path_m2", "pe_indirect",
        "pw_direct", "pw_path_m1", "pw_path_m2", "pw_indirect",
        "cs_direct", "cs_lumped_indirect",
        "nie",
    ]
    s3_recs = {k: [] for k in s3_keys}
    for r in range(cfg.n_runs):
        out = _run_s3(
            seed=r, sigma=0.5, n_total=cfg.n_total, n_eval=cfg.n_eval,
            model_type=cfg.model_type,
        )
        for k in s3_keys:
            s3_recs[k].append(out[k])
        print(f"  run {r + 1}/{cfg.n_runs}  "
              f"pe[m1,m2]=({out['pe_path_m1']:+.3f},"
              f"{out['pe_path_m2']:+.3f})")
    s3 = _mc_aggregate(s3_recs)

    return {
        "cfg": cfg,
        "s1": s1,
        "s2": s2,
        "s3": s3,
    }


def mc_bound(results):
    """Max Monte Carlo std across every reported cell."""
    cells = []
    for block in ("s1", "s2", "s3"):
        for (_, std) in results[block].values():
            cells.append(std)
    return max(cells)


# =============================================================================
# Pretty printers
# =============================================================================


def print_summary(results):
    cfg = results["cfg"]
    print("\n" + "=" * 68)
    print(f"Main-table results   (R={cfg.n_runs}, n_total={cfg.n_total}, "
          f"n_eval={cfg.n_eval}, model={cfg.model_type})")
    print("=" * 68)

    def fmt(pair):
        m, s = pair
        return f"{m:+.3f}  (mc_std={s:.3f})"

    s1, s2, s3 = results["s1"], results["s2"], results["s3"]

    print("\nS1  chain path  T -> M1 -> M2 -> Y")
    print(f"  PE-SHAP      = {fmt(s1['pe'])}")
    print(f"  PW-SHAP      = {fmt(s1['pw'])}")
    print(f"  Causal SHAP  = {fmt(s1['cs'])}")
    print(f"  NIE_Yhat     = {fmt(s1['nie'])}")

    print("\nS2  parallel with interaction")
    print("                  via M1    via M2    indirect")
    print(f"  PE-SHAP      "
          f"{s2['pe_path_m1'][0]:+8.3f} "
          f"{s2['pe_path_m2'][0]:+8.3f} "
          f"{s2['pe_indirect'][0]:+8.3f}")
    print(f"  PW-SHAP      "
          f"{s2['pw_path_m1'][0]:+8.3f} "
          f"{s2['pw_path_m2'][0]:+8.3f} "
          f"  ---")
    print(f"  Causal SHAP  "
          f"  ---       ---     "
          f"{s2['cs_indirect'][0]:+8.3f}")
    print(f"  NIE_Yhat     "
          f"  ---       ---     "
          f"{s2['nie'][0]:+8.3f}")

    print("\nS3  combined DAG  path decomposition")
    print("                  direct    via M1    via M2    indirect")
    print(f"  PE-SHAP      "
          f"{s3['pe_direct'][0]:+8.3f} "
          f"{s3['pe_path_m1'][0]:+8.3f} "
          f"{s3['pe_path_m2'][0]:+8.3f} "
          f"{s3['pe_indirect'][0]:+8.3f}")
    print(f"  PW-SHAP      "
          f"{s3['pw_direct'][0]:+8.3f} "
          f"{s3['pw_path_m1'][0]:+8.3f} "
          f"{s3['pw_path_m2'][0]:+8.3f} "
          f"  ---")
    print(f"  Causal SHAP  "
          f"{s3['cs_direct'][0]:+8.3f} "
          f"  ---       ---     "
          f"{s3['cs_lumped_indirect'][0]:+8.3f}")
    print(f"  NIE_Yhat     "
          f"  ---     "
          f"  ---     "
          f"  ---     "
          f"{s3['nie'][0]:+8.3f}")

    print(f"\nMax MC std across all reported cells: {mc_bound(results):.4f}")
    print("(Round this up to a clean value and use it as the caption bound.)")


def print_latex(results):
    cfg = results["cfg"]
    s1, s2, s3 = results["s1"], results["s2"], results["s3"]

    def v(pair):
        m, s = pair
        return f"${m:+.2f}" + r"{\scriptstyle\pm" + f"{s:.2f}" + r"}$"

    def vb(pair):
        m, s = pair
        return r"$\mathbf{" + f"{m:+.2f}" + r"}{\scriptstyle\pm" + f"{s:.2f}" + r"}$"

    print("\n" + "=" * 68)
    print("LaTeX fragment (paste into the thesis experiments section):")
    print("=" * 68)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l c | ccc | ccc}",
        r"\toprule",
        (r"& \textbf{S1}"
         r" & \multicolumn{3}{c|}{\textbf{S2}}"
         r" & \multicolumn{3}{c}{\textbf{S3}} \\"),
        (r"\cmidrule(lr){2-2}"
         r" \cmidrule(lr){3-5}"
         r" \cmidrule(lr){6-8}"),
        (r"\textbf{Met.}"
         r" & $M_1$"
         r" & $M_1$ & $M_2$ & $M_{1,2}$"
         r" & $M_1$ & $M_2$ & $M_{1,2}$ \\"),
        r"\midrule",
        # PE
        (f"PE"
         f" & {v(s1['pe'])}"
         f" & {v(s2['pe_path_m1'])} & {v(s2['pe_path_m2'])}"
         f" & {v(s2['pe_indirect'])}"
         f" & {v(s3['pe_path_m1'])}"
         f" & {v(s3['pe_path_m2'])} & {v(s3['pe_indirect'])} \\\\"),
        # PW
        (f"PW"
         f" & {vb(s1['pw'])}"
         f" & {v(s2['pw_path_m1'])} & {v(s2['pw_path_m2'])}"
         f" & {v(s2['pw_indirect'])}"
         f" & {vb(s3['pw_path_m1'])}"
         f" & {vb(s3['pw_path_m2'])} & {v(s3['pw_indirect'])} \\\\"),
        # CS
        (f"CS"
         f" & {v(s1['cs'])}"
         f" & --- & ---"
         f" & {v(s2['cs_indirect'])}"
         f" & --- & ---"
         f" & {v(s3['cs_lumped_indirect'])} \\\\"),
        r"\midrule",
        # NIE
        (f"$\\mathrm{{NIE}}_{{\\hat Y}}$"
         f" & {v(s1['nie'])}"
         f" & --- & ---"
         f" & {v(s2['nie'])}"
         f" & --- & ---"
         f" & {v(s3['nie'])} \\\\"),
        r"\bottomrule",
        r"\end{tabular}",
        (r"\caption{Path-level attributions across three synthetic benchmarks."
         f" Each cell: mean $\\pm$ std over $R = {cfg.n_runs}$ independent"
         r" runs (fresh data, retrained MLP)."
         r" $M_{{1,2}}$: total mediated effect."
         r" Bold: structural PW failures."
         r" CS and $\mathrm{NIE}_{\hat Y}$: total indirect only.}"),
        r"\label{tab:main-results}",
        r"\end{table*}",
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    cfg = RunConfig()
    results = run_all(cfg)
    print_summary(results)
    print_latex(results)
