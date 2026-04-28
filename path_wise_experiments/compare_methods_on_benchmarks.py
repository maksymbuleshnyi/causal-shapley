from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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


def _nie_s1(model, X_train, s1_params, n_samples=2000, seed=42):
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


def _cde_g_formula(model, X_train, T_col, ancestor_cols, fix_col, fix_value,
                   rng=None):
    """Plug-in g-computation CDE(X_{fix_col} = fix_value).

    Used when Property 1 fails: the fix variable has unblocked ancestors
    that need to be resampled from their T-conditional distributions to
    avoid reintroducing confounding through C.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(X_train)

    M_T1 = {a: X_train[X_train[:, T_col] == 1, a] for a in ancestor_cols}
    M_T0 = {a: X_train[X_train[:, T_col] == 0, a] for a in ancestor_cols}

    X_synth_1 = X_train.copy()
    X_synth_1[:, T_col] = 1
    for a in ancestor_cols:
        X_synth_1[:, a] = rng.choice(M_T1[a], size=n)
    X_synth_1[:, fix_col] = fix_value
    y_do_T1 = float(model.predict(X_synth_1).mean())

    X_synth_0 = X_train.copy()
    X_synth_0[:, T_col] = 0
    for a in ancestor_cols:
        X_synth_0[:, a] = rng.choice(M_T0[a], size=n)
    X_synth_0[:, fix_col] = fix_value
    y_do_T0 = float(model.predict(X_synth_0).mean())

    return y_do_T1 - y_do_T0


def _ate_g_formula(model, X_train, T_col):
    X1 = X_train.copy()
    X1[:, T_col] = 1
    X0 = X_train.copy()
    X0[:, T_col] = 0
    return float(model.predict(X1).mean() - model.predict(X0).mean())


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


# ---------------------------------------------------------------------------
# Per-benchmark single-run estimators
# ---------------------------------------------------------------------------


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

    # Property 1 fails for lambda_{M_2} on S3 (M_1 is an ancestor of M_2),
    # so the C-CATE estimator does not identify CDE(M_2 = m_2). Recover
    # lambda_{M_2} via the g-formula treating M_1 as an ancestor mediator
    # resampled from P(M_1 | T=t).
    ate_g = _ate_g_formula(model, X_train, COMB_T_COL)
    rng_g = np.random.default_rng(seed + 50_000)
    cde_m2_vals = []
    for x in ctrl:
        cde_m2_vals.append(
            _cde_g_formula(
                model, X_train, COMB_T_COL,
                ancestor_cols=(COMB_M1_COL,),
                fix_col=COMB_M2_COL, fix_value=x[COMB_M2_COL],
                rng=rng_g,
            )
        )
    pe_path_m2_g = ate_g - float(np.mean(cde_m2_vals))

    nie = _nie_s3(model, X_train)
    return {
        "pe_direct": pe["direct"],
        "pe_path_m1": pe["path_m1"],
        "pe_path_m2": pe_path_m2_g,
        "pe_indirect": s["causal_indirect"]["mean"],
        "pw_direct": pw["direct"],
        "pw_path_m1": pw["path_m1"],
        "pw_path_m2": pw["path_m2"],
        "pw_indirect": -s["causal_indirect"]["mean"],
        "cs_direct": s["direct_cde"]["mean"],
        "cs_lumped_indirect": s["causal_indirect"]["mean"],
        "nie": nie,
    }


# ---------------------------------------------------------------------------
# Monte Carlo aggregation
# ---------------------------------------------------------------------------


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
    # S2 Causal SHAP: full 2^n_features coalition sweep with MC draws per
    # coalition, so the inner-loop size is kept small.
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


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _cell(pair):
    m, s = pair
    return f"{m:+.3f} +/- {s:.3f}"


def _block(title, rows, width=72):
    lines = ['=' * width, title, '=' * width,
             f"{'Metric':<35}{'Mean':>14}{'Std':>14}",
             '-' * width]
    for name, (m, s) in rows:
        lines.append(f"{name:<35}{m:>+14.4f}{s:>14.4f}")
    return '\n'.join(lines)


def build_summary(results):
    cfg = results["cfg"]
    s1, s2, s3 = results["s1"], results["s2"], results["s3"]

    parts = []
    parts.append('=' * 80)
    parts.append("Path-wise attributions on S1, S2, S3")
    parts.append('=' * 80)
    parts.append(f"R = {cfg.n_runs} repetitions   "
                 f"n_total = {cfg.n_total}   "
                 f"n_eval = {cfg.n_eval}   "
                 f"model = {cfg.model_type}")
    parts.append('')

    parts.append(_block(
        "S1: chain mediation  (T -> M1 -> M2 -> Y)",
        [("PE-SHAP (M1 path)", s1["pe"]),
         ("PW-SHAP (M1 path)", s1["pw"]),
         ("Causal SHAP (indirect)", s1["cs"]),
         ("NIE_Yhat (reference)", s1["nie"])],
    ))
    parts.append('')

    parts.append(_block(
        "S2: parallel mediators with interaction",
        [("PE-SHAP direct", s2["pe_direct"]),
         ("PE-SHAP M1 path", s2["pe_path_m1"]),
         ("PE-SHAP M2 path", s2["pe_path_m2"]),
         ("PE-SHAP total mediated", s2["pe_indirect"]),
         ("PW-SHAP M1 path", s2["pw_path_m1"]),
         ("PW-SHAP M2 path", s2["pw_path_m2"]),
         ("Causal SHAP indirect (sum)", s2["cs_indirect"]),
         ("NIE_Yhat (reference)", s2["nie"])],
    ))
    parts.append('')

    parts.append(_block(
        "S3: combined realistic DAG",
        [("PE-SHAP direct", s3["pe_direct"]),
         ("PE-SHAP M1 path", s3["pe_path_m1"]),
         ("PE-SHAP M2 path (g-formula)", s3["pe_path_m2"]),
         ("PE-SHAP total mediated", s3["pe_indirect"]),
         ("PW-SHAP M1 path", s3["pw_path_m1"]),
         ("PW-SHAP M2 path", s3["pw_path_m2"]),
         ("Causal SHAP direct", s3["cs_direct"]),
         ("Causal SHAP lumped indirect", s3["cs_lumped_indirect"]),
         ("NIE_Yhat (reference)", s3["nie"])],
    ))
    parts.append('')

    # Combined layout: rows = methods, columns = (S1 M1) | (S2 M1, M2, M1,2) | (S3 M1, M2, M1,2)
    col_w = 8
    cell_w = 18
    total_w = col_w + 7 * cell_w
    header_top = (f"{'':<{col_w}}"
                  f"{'S1':^{cell_w}}"
                  f"{'S2':^{3 * cell_w}}"
                  f"{'S3':^{3 * cell_w}}")
    header_sub = (f"{'Method':<{col_w}}"
                  f"{'M1':^{cell_w}}"
                  f"{'M1':^{cell_w}}{'M2':^{cell_w}}{'M1,2':^{cell_w}}"
                  f"{'M1':^{cell_w}}{'M2':^{cell_w}}{'M1,2':^{cell_w}}")

    def row_method(label, s1_m1, s2_m1, s2_m2, s2_tot, s3_m1, s3_m2, s3_tot):
        return (f"{label:<{col_w}}"
                f"{_cell(s1_m1):^{cell_w}}"
                f"{_cell(s2_m1):^{cell_w}}{_cell(s2_m2):^{cell_w}}{_cell(s2_tot):^{cell_w}}"
                f"{_cell(s3_m1):^{cell_w}}{_cell(s3_m2):^{cell_w}}{_cell(s3_tot):^{cell_w}}")

    def row_dashes(label, s1_m1, s2_tot, s3_tot):
        return (f"{label:<{col_w}}"
                f"{_cell(s1_m1):^{cell_w}}"
                f"{'---':^{cell_w}}{'---':^{cell_w}}{_cell(s2_tot):^{cell_w}}"
                f"{'---':^{cell_w}}{'---':^{cell_w}}{_cell(s3_tot):^{cell_w}}")

    parts.append('=' * total_w)
    parts.append("Per-path attributions across S1, S2, S3 (mean +/- std)")
    parts.append('=' * total_w)
    parts.append(header_top)
    parts.append(header_sub)
    parts.append('-' * total_w)

    parts.append(row_method(
        "PE",
        s1["pe"],
        s2["pe_path_m1"], s2["pe_path_m2"], s2["pe_indirect"],
        s3["pe_path_m1"], s3["pe_path_m2"], s3["pe_indirect"],
    ))
    parts.append(row_method(
        "PW",
        s1["pw"],
        s2["pw_path_m1"], s2["pw_path_m2"], s2["pw_indirect"],
        s3["pw_path_m1"], s3["pw_path_m2"], s3["pw_indirect"],
    ))
    parts.append(row_dashes(
        "CS",
        s1["cs"], s2["cs_indirect"], s3["cs_lumped_indirect"],
    ))
    parts.append('-' * total_w)
    parts.append(row_dashes(
        "NIE",
        s1["nie"], s2["nie"], s3["nie"],
    ))
    parts.append('=' * total_w)
    parts.append('')
    parts.append(f"Max MC std across all reported cells: {mc_bound(results):.4f}")

    return '\n'.join(parts)


if __name__ == "__main__":
    cfg = RunConfig()
    results = run_all(cfg)

    summary = build_summary(results)
    print('\n' + summary)

    os.makedirs('./results', exist_ok=True)
    out_path = './results/methods_on_benchmarks.txt'
    with open(out_path, 'w') as f:
        f.write(summary + '\n')
    print(f"\nWrote summary to {out_path}")
