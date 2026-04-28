from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from data.simulations import (
    parallel_interaction_dataset,
    PAR_T_COL, PAR_M1_COL, PAR_M2_COL,
)
from path_wise_experiments._common import train_model, split_train_eval
from path_wise_experiments.path_wise.synthetic_attribution import (
    local_decomposition,
    cate_block,
)
from path_wise_experiments.path_wise.causal_shapley_local import (
    causal_shapley_direct_indirect_row,
)


S2_DAG_PARENTS = {
    PAR_T_COL: [],
    PAR_M1_COL: [PAR_T_COL],
    PAR_M2_COL: [PAR_T_COL],
}

S2_PARAMS = dict(
    alpha1=0.5, alpha2=0.7,
    beta1=0.3, beta2=0.2, beta3=-1.1,
    beta4=0.2, beta5=0.8,
)

# Sample 1: typical treated; Sample 2: high M1 — interaction flips M2 sign.
FIXED_SAMPLE_1 = np.array([1.0, 0.50, 0.70])
FIXED_SAMPLE_2 = np.array([1.0, 1.50, 0.70])


def _run_one(seed, sample_1, sample_2,
             n_total=10000, n_eval=200, model_type="mlp",
             cs_n_mc=200, cs_eps=0.25,
             eps_mediator=0.4, eps_base=0.5):
    X, y = parallel_interaction_dataset(
        n=n_total, sigma=0.5, seed=seed, **S2_PARAMS,
    )
    X_train, y_train, _, _ = split_train_eval(
        X, y, eval_size=n_eval, seed=seed,
    )
    model = train_model(X_train, y_train, model_type=model_type)

    out = {}
    for tag, sample in [("s1", sample_1), ("s2", sample_2)]:
        dec = local_decomposition(
            model, X_train, PAR_T_COL,
            [PAR_M1_COL, PAR_M2_COL], sample,
            base_cond_cols=(),
            eps_mediator=eps_mediator, eps_base=eps_base,
        )
        out[f"{tag}_pe_m1"] = dec["phi"][PAR_M1_COL]
        out[f"{tag}_pe_m2"] = dec["phi"][PAR_M2_COL]
        out[f"{tag}_pe_sum"] = dec["phi"][PAR_M1_COL] + dec["phi"][PAR_M2_COL]
        out[f"{tag}_lam_m1"] = dec["lam"][PAR_M1_COL]
        out[f"{tag}_lam_m2"] = dec["lam"][PAR_M2_COL]
        out[f"{tag}_lam_m1m2"] = dec["local_mediated"]
        out[f"{tag}_v_free"] = dec["v_free"]
        out[f"{tag}_v_all"] = dec["v_all"]
        out[f"{tag}_delta"] = dec["interaction_surplus"]

        v_all = cate_block(
            model, X_train, PAR_T_COL,
            [PAR_M1_COL, PAR_M2_COL],
            (), sample,
            eps_mediator=eps_mediator, eps_base=eps_base,
        )
        v_all_but_m1 = cate_block(
            model, X_train, PAR_T_COL,
            [PAR_M2_COL],
            (), sample,
            eps_mediator=eps_mediator, eps_base=eps_base,
        )
        v_all_but_m2 = cate_block(
            model, X_train, PAR_T_COL,
            [PAR_M1_COL],
            (), sample,
            eps_mediator=eps_mediator, eps_base=eps_base,
        )
        out[f"{tag}_pw_m1"] = v_all - v_all_but_m1
        out[f"{tag}_pw_m2"] = v_all - v_all_but_m2

        rng = np.random.default_rng(seed)
        phi_tot, phi_dir, phi_ind, _ = causal_shapley_direct_indirect_row(
            model, X_train, sample, S2_DAG_PARENTS,
            n_mc=cs_n_mc, eps=cs_eps, rng=rng,
        )
        out[f"{tag}_cs_dir_T"] = phi_dir[PAR_T_COL]
        out[f"{tag}_cs_ind_T"] = phi_ind[PAR_T_COL]
        out[f"{tag}_cs_tot_T"] = phi_tot[PAR_T_COL]

    return out


def run_all(n_runs=20, **kwargs):
    sample_1 = FIXED_SAMPLE_1
    sample_2 = FIXED_SAMPLE_2
    col_names = ["T", "M1", "M2"]
    print("  Fixed samples:")
    print("    s1: " + ", ".join(f"{col_names[i]}={sample_1[i]:.4f}" for i in range(3)))
    print("    s2: " + ", ".join(f"{col_names[i]}={sample_2[i]:.4f}" for i in range(3)))

    all_keys = None
    results = []
    for r in range(n_runs):
        print(f"  run {r+1}/{n_runs} ...", end=" ", flush=True)
        row = _run_one(seed=1000 + r, sample_1=sample_1,
                       sample_2=sample_2, **kwargs)
        print("done")
        results.append(row)
        if all_keys is None:
            all_keys = sorted(row.keys())

    agg = {}
    for k in all_keys:
        vals = np.array([r[k] for r in results])
        agg[k] = {"mean": float(vals.mean()), "std": float(vals.std())}
    agg["s1_sample"] = sample_1.copy()
    agg["s2_sample"] = sample_2.copy()
    agg["n_runs"] = n_runs
    return agg


def _analytic_gt():
    p = S2_PARAMS
    pe_m1 = p["alpha1"] * (p["beta2"] + p["beta4"] + p["alpha2"] * p["beta5"])
    pe_m2 = p["alpha2"] * (p["beta3"] + p["alpha1"] * p["beta5"])
    pw_m1 = p["alpha1"] * (p["beta2"] + p["beta4"])
    pw_m2 = p["alpha2"] * p["beta3"]
    return {"pe_m1": pe_m1, "pe_m2": pe_m2, "pw_m1": pw_m1, "pw_m2": pw_m2}


def _block(title, rows, width=72):
    lines = ['=' * width, title, '=' * width,
             f"{'Metric':<35}{'Mean':>14}{'Std':>14}",
             '-' * width]
    for name, m, s in rows:
        lines.append(f"{name:<35}{m:>+14.4f}{s:>14.4f}")
    return '\n'.join(lines)


def build_summary(agg):
    col_names = ["T", "M1", "M2"]
    parts = []
    parts.append('=' * 80)
    parts.append("Local attributions on S2 -- two treated samples")
    parts.append('=' * 80)
    parts.append(f"n_runs = {agg['n_runs']}   "
                 f"(mean +/- MC std over independent retrains)")
    parts.append('')

    for tag, label in [("s1", "Sample x_1 (typical treated)"),
                       ("s2", "Sample x_2 (high M1)")]:
        sample = agg[f"{tag}_sample"]
        feat_str = ", ".join(f"{col_names[i]}={sample[i]:.3f}"
                             for i in range(len(col_names)))
        rows = [
            ("PE lambda_M1",          agg[f"{tag}_lam_m1"]["mean"],   agg[f"{tag}_lam_m1"]["std"]),
            ("PE lambda_M2",          agg[f"{tag}_lam_m2"]["mean"],   agg[f"{tag}_lam_m2"]["std"]),
            ("PE lambda_M1M2",        agg[f"{tag}_lam_m1m2"]["mean"], agg[f"{tag}_lam_m1m2"]["std"]),
            ("PE phi_M1 (Shapley)",   agg[f"{tag}_pe_m1"]["mean"],    agg[f"{tag}_pe_m1"]["std"]),
            ("PE phi_M2 (Shapley)",   agg[f"{tag}_pe_m2"]["mean"],    agg[f"{tag}_pe_m2"]["std"]),
            ("PE sum(phi)",           agg[f"{tag}_pe_sum"]["mean"],   agg[f"{tag}_pe_sum"]["std"]),
            ("PW Psi_M1",             agg[f"{tag}_pw_m1"]["mean"],    agg[f"{tag}_pw_m1"]["std"]),
            ("PW Psi_M2",             agg[f"{tag}_pw_m2"]["mean"],    agg[f"{tag}_pw_m2"]["std"]),
            ("CS direct (T)",         agg[f"{tag}_cs_dir_T"]["mean"], agg[f"{tag}_cs_dir_T"]["std"]),
            ("CS indirect (T)",       agg[f"{tag}_cs_ind_T"]["mean"], agg[f"{tag}_cs_ind_T"]["std"]),
            ("CS total (T)",          agg[f"{tag}_cs_tot_T"]["mean"], agg[f"{tag}_cs_tot_T"]["std"]),
        ]
        parts.append(_block(f"{label}   ({feat_str})", rows))
        parts.append('')

    # Combined two-row table (one row per sample, columns = key methods).
    cell_w = 22
    col_w = 14
    total_w = col_w + 6 * cell_w

    def cell(key, tag):
        m, s = agg[f"{tag}_{key}"]["mean"], agg[f"{tag}_{key}"]["std"]
        return f"{m:+.3f} +/- {s:.3f}"

    header = (f"{'Sample':<{col_w}}"
              f"{'phi^PE_M1':^{cell_w}}{'phi^PE_M2':^{cell_w}}{'sum phi^PE':^{cell_w}}"
              f"{'Psi^PW_M1':^{cell_w}}{'Psi^PW_M2':^{cell_w}}{'phi^CS_T':^{cell_w}}")
    parts.append('=' * total_w)
    parts.append("Combined per-sample summary (mean +/- std across runs)")
    parts.append('=' * total_w)
    parts.append(header)
    parts.append('-' * total_w)
    for tag, label in [("s1", "x_1"), ("s2", "x_2")]:
        parts.append(
            f"{label:<{col_w}}"
            f"{cell('pe_m1', tag):^{cell_w}}"
            f"{cell('pe_m2', tag):^{cell_w}}"
            f"{cell('pe_sum', tag):^{cell_w}}"
            f"{cell('pw_m1', tag):^{cell_w}}"
            f"{cell('pw_m2', tag):^{cell_w}}"
            f"{cell('cs_tot_T', tag):^{cell_w}}"
        )
    parts.append('=' * total_w)

    gt = _analytic_gt()
    parts.append('')
    parts.append("Analytic ground truth (from S2 SCM, no data, no model):")
    parts.append(f"  PE path M1 = {gt['pe_m1']:+.4f}")
    parts.append(f"  PE path M2 = {gt['pe_m2']:+.4f}")
    parts.append(f"  PW path M1 = {gt['pw_m1']:+.4f}")
    parts.append(f"  PW path M2 = {gt['pw_m2']:+.4f}")

    scalar_keys = [k for k in agg
                   if not k.endswith("_sample") and k != "n_runs"]
    max_std = max(agg[k]["std"] for k in scalar_keys)
    parts.append('')
    parts.append(f"Max MC std across all quantities: {max_std:.4f}")

    return '\n'.join(parts)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-runs", type=int, default=20)
    p.add_argument("--n-total", type=int, default=10000)
    p.add_argument("--n-eval", type=int, default=200)
    p.add_argument("--model-type", default="mlp")
    p.add_argument("--cs-n-mc", type=int, default=200)
    p.add_argument("--cs-eps", type=float, default=0.25)
    args = p.parse_args()

    print(f"Running {args.n_runs} MC trials for local figure values ...")
    print(f"S2 params: {S2_PARAMS}")
    agg = run_all(
        n_runs=args.n_runs,
        n_total=args.n_total,
        n_eval=args.n_eval,
        model_type=args.model_type,
        cs_n_mc=args.cs_n_mc,
        cs_eps=args.cs_eps,
    )

    summary = build_summary(agg)
    print('\n' + summary)

    os.makedirs('./results', exist_ok=True)
    out_path = './results/local_attributions_s2.txt'
    with open(out_path, 'w') as f:
        f.write(summary + '\n')
    print(f"\nWrote summary to {out_path}")
