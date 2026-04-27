"""
Heskes-style symmetric Causal Shapley values for local explanations of
a trained black-box model on a known DAG.

Definitions.  For a feature set ``S``, the value function is

    v(S, x) = E[ f(X) | do(X_S = x_S) ]

i.e. the model's expected output when the features in ``S`` are fixed at
the sample's values and the remaining features are resampled from the
*do*-mutilated distribution induced by the DAG.  Once ``v(S, x)`` is in
hand for every subset ``S`` of the feature set, the Causal Shapley value
for feature ``i`` at the sample is the ordinary symmetric Shapley
aggregation

    phi_i(x) = sum_{S subset N \ {i}}
                  |S|! (n - |S| - 1)! / n!
                  * ( v(S ∪ {i}, x) − v(S, x) ).

By construction ``sum_i phi_i(x) = f(x) − v(∅, x)``, which is the usual
SHAP efficiency / local-accuracy identity.

The value function is estimated by Monte Carlo: we walk the DAG in
topological order; features in ``S`` are pinned at ``x`` and each
non-intervened feature is sampled from its conditional on its *already
filled* parents via a filter-close window on the training set.  Under
the chain- and parallel-SEM DAGs used in the thesis experiments this
produces an unbiased estimate of ``E[ f(X) | do(X_S = x_S) ]`` up to the
usual nonparametric-regression bias of the window.
"""

from __future__ import annotations

from itertools import combinations
from math import factorial

import numpy as np


def _topological_order(dag_parents, n_features):
    """Kahn-style topological sort of feature indices."""
    order = []
    remaining = set(range(n_features))
    while remaining:
        ready = sorted(
            f for f in remaining
            if set(dag_parents.get(f, [])).issubset(order)
        )
        if not ready:
            raise ValueError(
                "DAG has a cycle or references a parent not in the feature set."
            )
        order.extend(ready)
        remaining -= set(ready)
    return order


def _sample_conditional(X_train, feature, parents, current_row, eps, rng):
    """
    Draw ``X[feature]`` from the filter-close approximation of
    ``P(X[feature] | Pa(X[feature]) = current_row[parents])``.

    If ``parents`` is empty this collapses to a marginal draw from
    ``X_train[:, feature]``.  If the filter-close window is empty the
    nearest-neighbour row is used as a fallback so the Monte Carlo draw
    is always well defined.
    """
    if not parents:
        j = int(rng.integers(len(X_train)))
        return float(X_train[j, feature])
    mask = np.ones(len(X_train), dtype=bool)
    for p in parents:
        mask &= np.abs(X_train[:, p] - current_row[p]) < eps
    candidates = np.where(mask)[0]
    if candidates.size == 0:
        parent_vec = np.asarray([current_row[p] for p in parents], dtype=float)
        dists = np.linalg.norm(X_train[:, parents] - parent_vec, axis=1)
        j = int(np.argmin(dists))
    else:
        j = int(rng.choice(candidates))
    return float(X_train[j, feature])


def v_causal(model, X_train, sample, S, dag_parents,
             n_mc=400, eps=0.25, rng=None):
    """
    Monte Carlo estimate of ``v(S, x) = E[ f(X) | do(X_S = x_S) ]``.

    Generates ``n_mc`` synthetic rows by walking the DAG in topological
    order.  At each step, features in ``S`` are pinned at the sample's
    values and every other feature is sampled from its conditional on
    its already-filled parents via ``_sample_conditional``.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    sample = np.asarray(sample, dtype=float)
    n_features = X_train.shape[1]
    order = _topological_order(dag_parents, n_features)
    S_set = set(int(i) for i in S)
    synth = np.empty((n_mc, n_features), dtype=float)
    for k in range(n_mc):
        row = np.zeros(n_features, dtype=float)
        for feat in order:
            if feat in S_set:
                row[feat] = sample[feat]
            else:
                parents = list(dag_parents.get(feat, []))
                row[feat] = _sample_conditional(
                    X_train, feat, parents, row, eps, rng,
                )
        synth[k] = row
    preds = np.asarray(model.predict(synth), dtype=float).ravel()
    return float(preds.mean())


def v_direct_add(model, X_train, sample, S, i, dag_parents,
                 n_mc=400, eps=0.25, rng=None):
    """
    Monte Carlo estimate of the *direct-only* value function used by the
    Heskes direct / indirect decomposition.

    Informally: "the value of coalition ``S`` after we add feature ``i``
    but refuse to let ``i``'s intervention propagate to its descendants
    in the DAG".  Each synthetic row is drawn under ``do(X_S = x_S)`` in
    the usual topological-walk way — which includes sampling ``X_i``
    from its conditional on parents and cascading that sampled value
    through any descendants of ``i`` — and then the row entry for ``i``
    is overwritten with ``x_i`` just before passing to ``model.predict``.
    Descendants of ``i`` therefore carry the *pre-intervention* value of
    ``i`` that was produced during the topological walk, isolating the
    direct (non-propagating) contribution of ``i`` to the prediction.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    sample = np.asarray(sample, dtype=float)
    n_features = X_train.shape[1]
    order = _topological_order(dag_parents, n_features)
    S_set = set(int(j) for j in S)
    if i in S_set:
        raise ValueError("v_direct_add called with i already in S.")
    synth = np.empty((n_mc, n_features), dtype=float)
    for k in range(n_mc):
        row = np.zeros(n_features, dtype=float)
        for feat in order:
            if feat in S_set:
                row[feat] = sample[feat]
            else:
                parents = list(dag_parents.get(feat, []))
                row[feat] = _sample_conditional(
                    X_train, feat, parents, row, eps, rng,
                )
        row[i] = sample[i]
        synth[k] = row
    preds = np.asarray(model.predict(synth), dtype=float).ravel()
    return float(preds.mean())


def causal_shapley_direct_indirect_row(model, X_train, sample, dag_parents,
                                       n_mc=400, eps=0.25, rng=None):
    """
    Heskes direct / indirect decomposition of per-feature symmetric
    Causal Shapley values at a single sample.

    Returns
    -------
    phi_total : ndarray, shape (n_features,)
        Total Causal Shapley value per feature (same as what
        ``causal_shapley_row`` returns).
    phi_direct : ndarray, shape (n_features,)
        The component of each feature's Shapley value that does NOT
        propagate through descendants.  For a feature with no direct
        edge to the outcome, this is zero.
    phi_indirect : ndarray, shape (n_features,)
        The component that comes from the feature's intervention
        propagating through its descendants.  For a feature with no
        descendants in the DAG, this is zero.
    v_table : dict[frozenset[int], float]
        Full coalition value table.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    sample = np.asarray(sample, dtype=float)
    n = X_train.shape[1]
    features = list(range(n))

    v_table: dict[frozenset[int], float] = {}
    for k in range(n + 1):
        for S in combinations(features, k):
            v_table[frozenset(S)] = v_causal(
                model, X_train, sample, S, dag_parents,
                n_mc=n_mc, eps=eps, rng=rng,
            )

    phi_total = np.zeros(n, dtype=float)
    phi_direct = np.zeros(n, dtype=float)
    phi_indirect = np.zeros(n, dtype=float)
    n_fact = factorial(n)
    for i in features:
        others = [f for f in features if f != i]
        for k in range(len(others) + 1):
            for S in combinations(others, k):
                S_set = frozenset(S)
                w = factorial(k) * factorial(n - k - 1) / n_fact
                v_S = v_table[S_set]
                v_Si = v_table[S_set | {i}]
                delta_total = v_Si - v_S
                v_dir = v_direct_add(
                    model, X_train, sample, S_set, i, dag_parents,
                    n_mc=n_mc, eps=eps, rng=rng,
                )
                delta_direct = v_dir - v_S
                delta_indirect = delta_total - delta_direct
                phi_total[i] += w * delta_total
                phi_direct[i] += w * delta_direct
                phi_indirect[i] += w * delta_indirect
    return phi_total, phi_direct, phi_indirect, v_table


def causal_shapley_row(model, X_train, sample, dag_parents,
                       n_mc=400, eps=0.25, rng=None):
    """
    Per-feature symmetric Causal Shapley values at a single sample.

    Parameters
    ----------
    model
        Any trained regressor with a ``.predict(X)`` method returning a
        1-D array of outputs.
    X_train : ndarray, shape (N, n_features)
        Empirical distribution used for the do-sampler.
    sample : array-like, shape (n_features,)
        The point at which to compute the local Causal Shapley values.
    dag_parents : dict[int, list[int]]
        DAG adjacency.  ``dag_parents[i]`` is the list of parent indices
        of feature ``i``.  Root features should map to an empty list.
    n_mc : int
        Number of Monte Carlo draws used to estimate each ``v(S, x)``.
    eps : float
        Window half-width for the filter-close conditional sampler.
    rng : np.random.Generator, optional
        Source of randomness; a fresh generator with seed 0 is used if
        omitted.

    Returns
    -------
    phi : ndarray, shape (n_features,)
        Symmetric Causal Shapley values at ``sample``.  Satisfies
        ``phi.sum() ≈ f(sample) − v(∅, sample)`` up to Monte Carlo noise
        in the ``v(∅)`` estimate.
    v_table : dict[frozenset[int], float]
        The full per-coalition value table, returned for diagnostics
        (e.g. checking the efficiency identity or comparing against the
        closed-form SEM expression).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    sample = np.asarray(sample, dtype=float)
    n = X_train.shape[1]
    features = list(range(n))

    v_table: dict[frozenset[int], float] = {}
    for k in range(n + 1):
        for S in combinations(features, k):
            v_table[frozenset(S)] = v_causal(
                model, X_train, sample, S, dag_parents,
                n_mc=n_mc, eps=eps, rng=rng,
            )

    phi = np.zeros(n, dtype=float)
    n_fact = factorial(n)
    for i in features:
        others = [f for f in features if f != i]
        for k in range(len(others) + 1):
            for S in combinations(others, k):
                S_set = frozenset(S)
                w = factorial(k) * factorial(n - k - 1) / n_fact
                phi[i] += w * (v_table[S_set | {i}] - v_table[S_set])
    return phi, v_table
