"""
Clean path-attribution estimators for the synthetic PE-SHAP experiments.

All estimators work on a generic layout where each row of X contains a
binary treatment column, an optional confounder column, and one or more
mediator columns.  They evaluate attributions for a trained black-box
model by conditioning on subsets of features via a nearest-neighbour
window on X (the standard filter-close estimator used in the existing
``path_wise/path_wise.py`` module, specialised for arbitrary mediator
layouts).

Definitions (informal).  Let

    v_block(S)(x)
        = E[ f(T=1, x_S, X_{-S}) | window(x) ] - E[ f(T=0, x_S, X_{-S}) | window(x) ]

be the CATE conditional on the mediators in S being fixed at the
evaluation point x and the remaining mediators integrated out over
their empirical T-conditional distribution.  The paper-style
attributions are then

    PE-SHAP( T -> M_k -> Y )  =  v_block(emptyset)       -  v_block({M_k})
    PW-SHAP( T -> M_k -> Y )  =  v_block(all mediators)  -  v_block(all mediators \ {M_k})
    Causal indirect (lumped)  =  v_block(emptyset)       -  v_block(all mediators)
    Direct (CDE)              =  v_block(all mediators)

Averaging these over held-out samples drawn from the control arm
(T = 0) recovers the analytic path-effect formulas derived in the
thesis motivational examples.
"""

from __future__ import annotations

import numpy as np


def _window_mask(X, cond_cols, sample, eps):
    """Boolean mask of rows in X within ``eps`` of ``sample`` on ``cond_cols``."""
    mask = np.ones(len(X), dtype=bool)
    for c in cond_cols:
        mask &= np.abs(X[:, c] - sample[c]) < eps
    return mask


def cate_block(model, X, T_col, block_cols, base_cond_cols, sample,
               eps_mediator=0.2, eps_base=0.3):
    """
    Estimate v_block(block_cols)(sample) on a trained ``model``.

    ``base_cond_cols`` are columns that should always be conditioned on
    (e.g. the confounder in Experiment 3).  ``block_cols`` are the
    mediators to be fixed at their sample values.  All other mediators
    are integrated out via the T-conditional empirical distribution in
    the window.
    """
    cond_cols = list(base_cond_cols) + list(block_cols)
    if cond_cols:
        eps_vec = {}
        for c in cond_cols:
            eps_vec[c] = eps_base if c in base_cond_cols else eps_mediator
        mask = np.ones(len(X), dtype=bool)
        for c, e in eps_vec.items():
            mask &= np.abs(X[:, c] - sample[c]) < e
    else:
        mask = np.ones(len(X), dtype=bool)

    X_sub = X[mask]
    T_sub = X_sub[:, T_col]
    X_t1 = X_sub[T_sub == 1]
    X_t0 = X_sub[T_sub == 0]
    if len(X_t1) == 0 or len(X_t0) == 0:
        return np.nan

    y1 = model.predict(X_t1).mean()
    y0 = model.predict(X_t0).mean()
    return y1 - y0


def attribution_row(model, X, T_col, mediator_cols, sample,
                    base_cond_cols=(),
                    eps_mediator=0.2, eps_base=0.3):
    """
    Compute the full attribution row (PE, PW, causal-style) for a single
    sample and return a dict keyed by descriptive names.

    For each mediator M_k the returned dict contains:

        pe_t_mk_y  =  v_block(emptyset)  -  v_block({M_k})
        pw_t_mk_y  =  v_block(all)       -  v_block(all \ {M_k})

    Plus:

        direct_cde    =  v_block(all)
        total_te      =  v_block(emptyset)
        causal_indirect =  total_te - direct_cde
    """
    base_cond_cols = list(base_cond_cols)
    all_mediators = list(mediator_cols)

    v_free = cate_block(model, X, T_col, [], base_cond_cols, sample,
                        eps_mediator, eps_base)
    v_all = cate_block(model, X, T_col, all_mediators, base_cond_cols,
                       sample, eps_mediator, eps_base)

    out = {
        "total_te": v_free,
        "direct_cde": v_all,
        "causal_indirect": v_free - v_all if not np.isnan(v_free) and not np.isnan(v_all) else np.nan,
    }

    v_block_one = {}
    for m in all_mediators:
        v_block_one[m] = cate_block(model, X, T_col, [m], base_cond_cols,
                                    sample, eps_mediator, eps_base)

    v_block_all_but_one = {}
    for m in all_mediators:
        others = [x for x in all_mediators if x != m]
        v_block_all_but_one[m] = cate_block(model, X, T_col, others,
                                            base_cond_cols, sample,
                                            eps_mediator, eps_base)

    for m in all_mediators:
        out[f"pe_m{m}"] = v_free - v_block_one[m]
        out[f"pw_m{m}"] = v_all - v_block_all_but_one[m]

    return out


def attribution_average(model, X, T_col, mediator_cols, X_eval,
                        base_cond_cols=(),
                        eps_mediator=0.2, eps_base=0.3):
    """
    Compute attribution_row for every sample in ``X_eval`` and return
    per-key arrays of values (one per sample, NaNs dropped in
    ``attribution_summary`` below).
    """
    rows = []
    for i in range(len(X_eval)):
        rows.append(attribution_row(
            model, X, T_col, mediator_cols, X_eval[i],
            base_cond_cols=base_cond_cols,
            eps_mediator=eps_mediator, eps_base=eps_base,
        ))
    keys = rows[0].keys()
    out = {k: np.array([r[k] for r in rows], dtype=float) for k in keys}
    return out


def local_decomposition(model, X, T_col, mediator_cols, sample,
                        base_cond_cols=(),
                        eps_mediator=0.2, eps_base=0.3):
    """
    Single-sample local PE-SHAP decomposition for the horizontal-bar figure.

    Returns a dict containing, all evaluated locally around ``sample``:

        v_free            = v_block(emptyset)(x)            local total effect
        v_all             = v_block(all mediators)(x)       local direct (CDE)
        v_one[m]          = v_block({m})(x)                 for each mediator m
        lam[m]            = v_free - v_one[m]               raw path-wise PE
        phi[m]            = lam[m] - delta / K              Shapley-decomposed PE
        local_mediated    = v_free - v_all                  local total indirect
        interaction_surplus = sum(lam) - local_mediated     Moebius overshoot

    With two mediators this yields ``phi[M1] + phi[M2] = local_mediated``
    exactly, recovering the Shapley efficiency property pointwise.
    """
    mediators = list(mediator_cols)
    v_free = cate_block(model, X, T_col, [], base_cond_cols, sample,
                        eps_mediator, eps_base)
    v_all = cate_block(model, X, T_col, mediators, base_cond_cols, sample,
                       eps_mediator, eps_base)
    v_one = {
        m: cate_block(model, X, T_col, [m], base_cond_cols, sample,
                      eps_mediator, eps_base)
        for m in mediators
    }
    lam = {m: v_free - v_one[m] for m in mediators}
    local_mediated = v_free - v_all
    raw_sum = sum(lam.values())
    delta = raw_sum - local_mediated
    k = len(mediators)
    phi = {m: lam[m] - delta / k for m in mediators}
    return {
        "v_free": v_free,
        "v_all": v_all,
        "v_one": v_one,
        "lam": lam,
        "phi": phi,
        "local_mediated": local_mediated,
        "interaction_surplus": delta,
        "sample": np.asarray(sample).copy(),
    }


def pick_median_control_sample(X, T_col, mediator_cols):
    """
    Pick the control-arm (T=0) row whose mediator vector is closest to the
    control-arm marginal median.  Returns ``(index_in_X, sample_row)``.
    Used to anchor single-sample local explanations at a "representative"
    instance so the reported bar values are not dominated by one outlier.
    """
    mediators = list(mediator_cols)
    T = X[:, T_col]
    ctrl_idx = np.where(T == 0)[0]
    if len(ctrl_idx) == 0:
        raise ValueError("No control-arm samples in X.")
    ctrl = X[ctrl_idx][:, mediators]
    med = np.median(ctrl, axis=0)
    dists = np.linalg.norm(ctrl - med, axis=1)
    j = int(np.argmin(dists))
    return int(ctrl_idx[j]), X[ctrl_idx[j]].copy()


def attribution_summary(per_sample_dict):
    """Mean / std ignoring NaNs for each key in ``per_sample_dict``."""
    summary = {}
    for k, arr in per_sample_dict.items():
        valid = arr[~np.isnan(arr)]
        summary[k] = {
            "mean": float(valid.mean()) if len(valid) else float("nan"),
            "std": float(valid.std()) if len(valid) else float("nan"),
            "n": int(len(valid)),
        }
    return summary
