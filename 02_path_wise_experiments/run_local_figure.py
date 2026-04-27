"""
Compute local attribution values for the S2 bar-chart figure.

Uses the parallel-mediator-with-interaction DGP (Experiment 2):
    T ~ Bernoulli(0.5)
    M1 = α1·T + U1
    M2 = α2·T + U2
    Y  = β1·T + β2·M1 + β3·M2 + β4·M1·T + β5·M1·M2 + UY

Produces numbers for two treated-arm samples across three methods
(PE-SHAP, PW-SHAP, Causal Shapley direct/indirect).  Each run uses
an independently seeded dataset + MLP; results are averaged over
``n_runs`` Monte Carlo trials to report mean ± MC std.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from simulations.simulations import (
    parallel_interaction_dataset,
    PAR_T_COL, PAR_M1_COL, PAR_M2_COL,
)
from experiments._common import train_model, split_train_eval
from path_wise.synthetic_attribution import (
    local_decomposition,
    cate_block,
)
from path_wise.causal_shapley_local import causal_shapley_direct_indirect_row


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

# Columns: [T, M1, M2]
# S1: typical treated (T=1, near treated-arm means)
# S2: high-M1 treated (T=1, M1 well above mean — M2 sign flips)
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
    return agg


def print_results(agg):
    print("\n" + "=" * 72)
    print("LOCAL FIGURE VALUES  (mean ± MC std over independent runs)")
    print("=" * 72)
    col_names = ["T", "M1", "M2"]
    for tag, label in [("s1", "Sample 1 (typical treated)"),
                       ("s2", "Sample 2 (high M1)")]:
        print(f"\n--- {label} ---")
        s = agg[f"{tag}_sample"]
        feat_str = ", ".join(
            f"{col_names[i]}={s[i]:.3f}" for i in range(len(col_names))
        )
        print(f"  Features: {feat_str}")
        print(f"  PE-SHAP (raw path effects):")
        print(f"    lam_M1      = {agg[f'{tag}_lam_m1']['mean']:+.4f} ± {agg[f'{tag}_lam_m1']['std']:.4f}")
        print(f"    lam_M2      = {agg[f'{tag}_lam_m2']['mean']:+.4f} ± {agg[f'{tag}_lam_m2']['std']:.4f}")
        print(f"    lam_M1M2    = {agg[f'{tag}_lam_m1m2']['mean']:+.4f} ± {agg[f'{tag}_lam_m1m2']['std']:.4f}")
        print(f"    v_free      = {agg[f'{tag}_v_free']['mean']:+.4f} ± {agg[f'{tag}_v_free']['std']:.4f}")
        print(f"    v_all       = {agg[f'{tag}_v_all']['mean']:+.4f} ± {agg[f'{tag}_v_all']['std']:.4f}")
        print(f"    delta       = {agg[f'{tag}_delta']['mean']:+.4f} ± {agg[f'{tag}_delta']['std']:.4f}")
        print(f"  PE-SHAP (Shapley-corrected):")
        print(f"    phi_M1      = {agg[f'{tag}_pe_m1']['mean']:+.4f} ± {agg[f'{tag}_pe_m1']['std']:.4f}")
        print(f"    phi_M2      = {agg[f'{tag}_pe_m2']['mean']:+.4f} ± {agg[f'{tag}_pe_m2']['std']:.4f}")
        print(f"    sum(phi)    = {agg[f'{tag}_pe_sum']['mean']:+.4f} ± {agg[f'{tag}_pe_sum']['std']:.4f}")

        print(f"  PW-SHAP:")
        print(f"    Psi_M1      = {agg[f'{tag}_pw_m1']['mean']:+.4f} ± {agg[f'{tag}_pw_m1']['std']:.4f}")
        print(f"    Psi_M2      = {agg[f'{tag}_pw_m2']['mean']:+.4f} ± {agg[f'{tag}_pw_m2']['std']:.4f}")

        print(f"  Causal Shapley (T):")
        print(f"    phi_dir_T   = {agg[f'{tag}_cs_dir_T']['mean']:+.4f} ± {agg[f'{tag}_cs_dir_T']['std']:.4f}")
        print(f"    phi_ind_T   = {agg[f'{tag}_cs_ind_T']['mean']:+.4f} ± {agg[f'{tag}_cs_ind_T']['std']:.4f}")
        print(f"    phi_tot_T   = {agg[f'{tag}_cs_tot_T']['mean']:+.4f} ± {agg[f'{tag}_cs_tot_T']['std']:.4f}")
        print(f"    dir+ind gap = {agg[f'{tag}_cs_dir_T']['mean'] + agg[f'{tag}_cs_ind_T']['mean'] - agg[f'{tag}_cs_tot_T']['mean']:+.4f}")

    print("\n" + "=" * 72)
    print("LaTeX mock values (paste into figure):")
    print("=" * 72)
    def v(key):
        return f"{agg[key]['mean']:+.3f} ± {agg[key]['std']:.3f}"
    for tag, label in [("s1", "Sample x_1"), ("s2", "Sample x_2")]:
        s = agg[f"{tag}_sample"]
        feat_str = ", ".join(
            f"{col_names[i]}={s[i]:.3f}" for i in range(len(col_names))
        )
        print(f"\n% {label}  ({feat_str})")
        print(f"% lam_M1       = {v(f'{tag}_lam_m1')}")
        print(f"% lam_M2       = {v(f'{tag}_lam_m2')}")
        print(f"% lam_M1M2     = {v(f'{tag}_lam_m1m2')}")
        print(f"% delta        = {v(f'{tag}_delta')}")
        print(f"% phi^PE_M1    = {v(f'{tag}_pe_m1')}")
        print(f"% phi^PE_M2    = {v(f'{tag}_pe_m2')}")
        print(f"% sum phi^PE   = {v(f'{tag}_pe_sum')}")
        print(f"% Psi^PW_M1    = {v(f'{tag}_pw_m1')}")
        print(f"% Psi^PW_M2    = {v(f'{tag}_pw_m2')}")
        print(f"% phi^CS,dir_T = {v(f'{tag}_cs_dir_T')}")
        print(f"% phi^CS,ind_T = {v(f'{tag}_cs_ind_T')}")
        print(f"% phi^CS_T     = {v(f'{tag}_cs_tot_T')}")

    gt = _analytic_gt()
    print(f"\n% Analytic GT (PE path M1) = {gt['pe_m1']:+.4f}")
    print(f"% Analytic GT (PE path M2) = {gt['pe_m2']:+.4f}")
    print(f"% Analytic GT (PW path M1) = {gt['pw_m1']:+.4f}")
    print(f"% Analytic GT (PW path M2) = {gt['pw_m2']:+.4f}")

    scalar_keys = [k for k in agg if not k.endswith("_sample")]
    max_std = max(agg[k]["std"] for k in scalar_keys)
    print(f"\n% Max MC std across all quantities: {max_std:.4f}")


def _analytic_gt():
    p = S2_PARAMS
    pe_m1 = p["alpha1"] * (p["beta2"] + p["beta4"] + p["alpha2"] * p["beta5"])
    pe_m2 = p["alpha2"] * (p["beta3"] + p["alpha1"] * p["beta5"])
    pw_m1 = p["alpha1"] * (p["beta2"] + p["beta4"])
    pw_m2 = p["alpha2"] * p["beta3"]
    return {"pe_m1": pe_m1, "pe_m2": pe_m2, "pw_m1": pw_m1, "pw_m2": pw_m2}


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
    print_results(agg)
