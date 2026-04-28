import os
import time
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


def model_outcomes_coal(model, coal, local_obs, ref_point, imp=None,
                        predict_proba=False):
    """Vectorized version. Returns a (n_obs, n) array of predictions."""
    n_obs, d = np.shape(local_obs)
    n = len(ref_point)

    coal = np.asarray(coal, dtype=float)
    local_obs = np.asarray(local_obs, dtype=float)

    # Build the input once: (n_obs, d) where present features = local_obs and
    # absent features = nan. The loop over reference points only changed via
    # imp.transform's posterior sampling, so we tile once and transform once.
    base = local_obs.copy()
    base[:, coal == 0] = np.nan

    # Tile to (n_obs * n, d) so the imputer (with sample_posterior=True)
    # produces n independent samples for each of the n_obs test points.
    tiled = np.tile(base, (n, 1))

    nan_cols = np.isnan(tiled).any(axis=0)
    if not nan_cols.any() or imp is None:
        # No imputation needed, or no imputer given — fall back to bootstrap
        # sampling from the reference set for the absent columns.
        if nan_cols.any():
            idx = np.random.randint(n, size=tiled.shape[0])
            sampled = ref_point[idx]
            tiled[:, nan_cols] = sampled[:, nan_cols]
        filled = tiled
    else:
        filled = imp.transform(tiled)

    if predict_proba:
        preds = model.predict_proba(filled)[:, 1]
    else:
        preds = model.predict(filled)

    return preds.reshape(n, n_obs).T  # -> (n_obs, n)


class IterativeImputerSubsets:
    def __init__(self, subsets=(), **kwargs):
        self.imps = [IterativeImputer(**kwargs) for _ in subsets]
        self.subsets = list(subsets)

    def fit(self, x_train):
        for imp, subset in zip(self.imps, self.subsets):
            imp.fit(x_train[:, subset])

    def transform(self, x):
        x_new = x
        for imp, subset in zip(self.imps, self.subsets):
            x_new[:, subset] = imp.transform(x[:, subset])
        return x_new


def decorrelate(x, mask):
    out = x.copy()
    idx = np.random.randint(len(x), size=len(x))
    sampled = x[idx]
    out[:, mask] = sampled[:, mask]
    return out


def generate_data(case, N):
    t = np.random.binomial(n=1, p=0.5 * np.ones(N))

    if case == 'none':
        q = np.random.uniform(size=N)
        d = np.random.binomial(n=1, p=0.5, size=N)
    elif case == 'q_only':
        q = np.random.uniform(size=N)
        q = t * (3 / 5) * q + (1 - t) * ((3 / 5) * q + 2 / 5)
        d = np.random.binomial(n=1, p=0.5, size=N)
    elif case == 'd_only':
        q = np.random.uniform(size=N)
        d = np.random.binomial(n=1, p=4 / 5 - 3 / 5 * t)
    elif case == 'both':
        q = np.random.uniform(size=N)
        q = t * (3 / 5) * q + (1 - t) * ((3 / 5) * q + 2 / 5)
        d = np.random.binomial(n=1, p=4 / 5 - 3 / 5 * t)
    else:
        raise ValueError(f"unknown case {case!r}")

    # Outcome matches the thesis description: Y = C_1 + C_2
    y = q + d
    return t, q, d, y


def run_one_iteration(case, N=200, imp_max_iter=10):
    t, q, d, y = generate_data(case, N)

    x = pd.DataFrame({'q': q, 'd': d, 't': t})
    N_train = N // 2
    x_train, y_train = x.iloc[:N_train], y[:N_train]
    x_test = x.iloc[N_train:]

    if y_train.std() == 0:
        return None

    # Imputers (lowered max_iter)
    imp = IterativeImputer(max_iter=imp_max_iter, random_state=0,
                           sample_posterior=True)
    imp.fit(x_train.values)

    outcome = Pipeline([('poly', PolynomialFeatures(2)),
                        ('lr', LinearRegression())])
    outcome.fit(x_train, y_train)

    xtr = x_train.values
    xte = x_test.values


    def cate_given_S(S_mask_qd):
        """S_mask_qd: length-2 bool array for (q, d).
        Returns per-sample CATE conditioning on those features."""
        coal = np.array([S_mask_qd[0], S_mask_qd[1], 1])  # T treated as in-coal,
                                                          # value forced via xte.

        xte_t1 = xte.copy(); xte_t1[:, 2] = 1
        preds_t1 = model_outcomes_coal(outcome, coal, xte_t1, xtr, imp=imp).mean(axis=1)

        xte_t0 = xte.copy(); xte_t0[:, 2] = 0
        preds_t0 = model_outcomes_coal(outcome, coal, xte_t0, xtr, imp=imp).mean(axis=1)

        return preds_t1 - preds_t0

    cate_full  = cate_given_S([1, 1])  # condition on (q, d)
    cate_q     = cate_given_S([1, 0])  # condition on q only
    cate_d     = cate_given_S([0, 1])  # condition on d only

    # Path-wise effects (per-sample):
    psi_q_per_sample = cate_full - cate_d   # adding q to {d}
    psi_d_per_sample = cate_full - cate_q   # adding d to {q}

    psi_q_ratio = np.abs(psi_q_per_sample).mean()
    psi_d_ratio = np.abs(psi_d_per_sample).mean()

    # Causal Shapley setup
    imp_no_t_full = IterativeImputerSubsets([[True, True, False]],
                                            max_iter=imp_max_iter,
                                            random_state=0,
                                            sample_posterior=True)
    imp_no_t_full.fit(xtr)

    x_do_q = decorrelate(xtr, [True, False, False])
    imp_do_q = IterativeImputer(max_iter=imp_max_iter, random_state=0, sample_posterior=True); imp_do_q.fit(x_do_q)
    imp_do_q_no_t = IterativeImputerSubsets([[True, True, False]], max_iter=imp_max_iter, random_state=0, sample_posterior=True); imp_do_q_no_t.fit(x_do_q)

    x_do_d = decorrelate(xtr, [False, True, False])
    imp_do_d = IterativeImputer(max_iter=imp_max_iter, random_state=0, sample_posterior=True); imp_do_d.fit(x_do_d)
    imp_do_d_no_t = IterativeImputerSubsets([[True, True, False]], max_iter=imp_max_iter, random_state=0, sample_posterior=True); imp_do_d_no_t.fit(x_do_d)

    # Per-sample Causal Shapley (return shape (n_obs,) for direct/indirect),
    # so we can take |.| per sample before averaging across the test set.
    def cs_terms(coal_with_t, coal_without_t, imp_a, imp_b):
        a = model_outcomes_coal(outcome, coal_with_t,    xte, xtr, imp=imp_a).mean(axis=1)
        b = model_outcomes_coal(outcome, coal_without_t, xte, xtr, imp=imp_a).mean(axis=1)
        c = model_outcomes_coal(outcome, coal_with_t,    xte, xtr, imp=imp_b).mean(axis=1)
        return c - b, a - c

    full_dir,  full_ind  = cs_terms([1, 1, 1], [1, 1, 0], None, None)
    qcoal_dir, qcoal_ind = cs_terms([1, 0, 1], [1, 0, 0], imp_do_q, imp_do_q_no_t)
    dcoal_dir, dcoal_ind = cs_terms([0, 1, 1], [0, 1, 0], imp_do_d, imp_do_d_no_t)
    empty_dir, empty_ind = cs_terms([0, 0, 1], [0, 0, 0], imp,      imp_no_t_full)

    phi_direct   = 1/3*full_dir   + 1/6*qcoal_dir   + 1/6*dcoal_dir   + 1/3*empty_dir
    phi_indirect = 1/3*full_ind   + 1/6*qcoal_ind   + 1/6*dcoal_ind   + 1/3*empty_ind

    return {
        'psi_q': psi_q_ratio,
        'psi_d': psi_d_ratio,
        'phi_direct':   np.abs(phi_direct).mean(),
        'phi_indirect': np.abs(phi_indirect).mean(),
    }


def run_case(case, label, n_iterations=3, N=200, imp_max_iter=10):
    print('\n' + '=' * 72)
    print(f'Case: {label}')
    print('=' * 72)
    rows = []
    for it in range(n_iterations):
        t0 = time.time()
        result = run_one_iteration(case, N=N, imp_max_iter=imp_max_iter)
        if result is None:
            print(f'  iter {it}: degenerate draw, skipped')
            continue
        rows.append(result)
        print(f'  iter {it}: psi_q={result["psi_q"]:.4f}  '
              f'psi_d={result["psi_d"]:.4f}  '
              f'dir={result["phi_direct"]:.4f}  '
              f'ind={result["phi_indirect"]:.4f}  '
              f'({time.time()-t0:.1f}s)')

    df = pd.DataFrame(rows)
    return {
        'case': label,
        'psi_q_mean':       df['psi_q'].mean(),       'psi_q_std':       df['psi_q'].std(),
        'psi_d_mean':       df['psi_d'].mean(),       'psi_d_std':       df['psi_d'].std(),
        'phi_direct_mean':  df['phi_direct'].mean(),  'phi_direct_std':  df['phi_direct'].std(),
        'phi_indirect_mean':df['phi_indirect'].mean(),'phi_indirect_std':df['phi_indirect'].std(),
        'n_iter': len(df),
    }


def format_case_block(s):
    lines = []
    lines.append('=' * 72)
    lines.append(f"Case: {s['case']}   (iterations: {s['n_iter']})")
    lines.append('=' * 72)
    lines.append(f"{'Metric':<35}{'Mean':>14}{'Std':>14}")
    lines.append('-' * 72)
    pairs = [
        ('|Psi_{T -> C_1 -> Y}|',          s['psi_q_mean'],        s['psi_q_std']),
        ('|Psi_{T -> C_2 -> Y}|',          s['psi_d_mean'],        s['psi_d_std']),
        ('|phi_direct (Causal SHAP)|',     s['phi_direct_mean'],   s['phi_direct_std']),
        ('|phi_indirect (Causal SHAP)|',   s['phi_indirect_mean'], s['phi_indirect_std']),
    ]
    for name, m, st in pairs:
        lines.append(f"{name:<35}{m:>14.4f}{st:>14.4f}")
    return '\n'.join(lines)


def format_table(summaries):
    def cell(m, s):
        return f"{m:.3f} +/- {s:.3f}"

    col_w = 22
    cell_w = 18
    lines = []
    total_w = col_w + 4 * cell_w
    lines.append('=' * total_w)
    lines.append('Path-Wise vs Causal Shapley across mediation cases')
    lines.append('=' * total_w)
    header = (f"{'Case':<{col_w}}"
              f"{'|Psi_C1|':>{cell_w}}"
              f"{'|Psi_C2|':>{cell_w}}"
              f"{'|phi_dir|':>{cell_w}}"
              f"{'|phi_ind|':>{cell_w}}")
    lines.append(header)
    lines.append('-' * total_w)
    for s in summaries:
        lines.append(
            f"{s['case']:<{col_w}}"
            f"{cell(s['psi_q_mean'],        s['psi_q_std']):>{cell_w}}"
            f"{cell(s['psi_d_mean'],        s['psi_d_std']):>{cell_w}}"
            f"{cell(s['phi_direct_mean'],   s['phi_direct_std']):>{cell_w}}"
            f"{cell(s['phi_indirect_mean'], s['phi_indirect_std']):>{cell_w}}"
        )
    lines.append('=' * total_w)
    return '\n'.join(lines)


def main():
    np.random.seed(0)
    os.makedirs('./results', exist_ok=True)

    cases = [
        ('none',   'C_1, C_2 (no mediators)'),
        ('q_only', 'C_1-M, C_2 (only C_1 mediates)'),
        ('d_only', 'C_1, C_2-M (only C_2 mediates)'),
        ('both',   'C_1-M, C_2-M (both mediate)'),
    ]

    summaries, blocks = [], []
    t_start = time.time()
    for case, label in cases:
        s = run_case(case, label, n_iterations=20, N=500, imp_max_iter=10)
        summaries.append(s)
        blocks.append(format_case_block(s))

    table = format_table(summaries)

    full_report = '\n\n'.join(blocks) + '\n\n' + table + '\n'
    print('\n' + table)
    print(f"\nTotal wall time: {time.time() - t_start:.1f}s")

    out_path = './results/mediation_structures.txt'
    with open(out_path, 'w') as f:
        f.write(full_report)
    print(f"Wrote combined report to {out_path}")


if __name__ == '__main__':
    main()
