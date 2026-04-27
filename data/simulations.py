import numpy as np

# ============================================================
# Experiment 3 (legacy name) — parallel mediators with quadratic M1
# (kept unchanged for backward-compat with existing notebooks)
# ============================================================

def path_wise_dataset_1(num_samples=500, seed=0):
    np.random.seed(seed)
    features, prediction = [], []
    for _ in range(num_samples):
        confounder = np.random.normal(loc=0.3, scale=0.5)
        treatment = np.random.binomial(n=1, p=np.clip(0.8 - confounder, 0, 1))
        mediator1 = 0.5 * treatment + np.random.normal(loc=0.0, scale=0.5)
        mediator2 = 0.7 * treatment + np.random.normal(loc=0.0, scale=0.5)
        outcome = confounder + treatment - 0.8 * mediator1 ** 2 - 0.5 * treatment * mediator1
        features.append([treatment, confounder, mediator1, mediator2])
        prediction.append(outcome)

    return np.array(features), np.array(prediction)


treatment_col_index = 0
confounder_col_index = 1
mediator1_col_index = 2
mediator2_col_index = 3


class ModelWrapper():
    def predict(self, data):
        results = []
        for sample in data:
            treatment = sample[0]
            confounder = sample[1]
            mediator1 = sample[2]
            mediator2 = sample[3]
            results.append(confounder + treatment - 0.8 * mediator1 ** 2 - 0.5 * treatment * mediator1)
        return np.array(results)


def calculate_true_cate_but_mediator2(sample):
    """
    Calculate the true CATE without mediator2 for a given sample.

    Based on the true model:
    outcome = confounder + treatment - 0.8 * mediator1^2 + 0.7 * mediator2^2 - 0.5 * treatment * mediator1

    When mediator2 is removed, we need to calculate:
    E[Y(1) | confounder, mediator1] - E[Y(0) | confounder, mediator1]

    Where mediator2 is integrated out:
    E[mediator2^2 | treatment, confounder]
    """
    treatment = sample[0]
    confounder = sample[1]
    mediator1 = sample[2]

    E_mediator2_squared_t1 = 0.25 + (0.7 * 1) ** 2
    E_mediator2_squared_t0 = 0.25 + (0.7 * 0) ** 2

    cate_true = (confounder + 1 - 0.8 * mediator1**2 + 0.7 * E_mediator2_squared_t1 - 0.5 * 1 * mediator1) - \
                (confounder + 0 - 0.8 * mediator1**2 + 0.7 * E_mediator2_squared_t0 - 0.5 * 0 * mediator1)

    return cate_true


# ============================================================
# Experiment 1 — Chain mediation
#
#   T  ~ Bernoulli(0.5)
#   M1 = a1 * T + U1,   U1 ~ N(0, sigma^2)
#   M2 = a2 * M1 + U2,  U2 ~ N(0, sigma^2)
#   Y  = b1 * T + b2 * M2 + UY
#
# Columns: [T, M1, M2]. No confounder.
# Ground-truth chain-path effect (T -> M1 -> M2 -> Y) = a1 * a2 * b2.
# ============================================================

CHAIN_T_COL = 0
CHAIN_M1_COL = 1
CHAIN_M2_COL = 2


def chain_mediation_dataset(n=5000, a1=0.8, a2=0.7, b1=0.5, b2=1.0,
                            sigma=0.5, seed=0):
    rng = np.random.default_rng(seed)
    T = rng.binomial(1, 0.5, size=n).astype(float)
    U1 = rng.normal(0.0, sigma, size=n)
    U2 = rng.normal(0.0, sigma, size=n)
    UY = rng.normal(0.0, sigma, size=n)
    M1 = a1 * T + U1
    M2 = a2 * M1 + U2
    Y = b1 * T + b2 * M2 + UY
    X = np.column_stack([T, M1, M2])
    return X, Y


class ChainModelWrapper:
    """Closed-form SEM evaluation: Y = b1*T + b2*M2 (noise-free mean)."""
    def __init__(self, b1=0.5, b2=1.0):
        self.b1 = b1
        self.b2 = b2

    def predict(self, X):
        X = np.asarray(X)
        return self.b1 * X[:, CHAIN_T_COL] + self.b2 * X[:, CHAIN_M2_COL]


def chain_ground_truth(a1=0.8, a2=0.7, b1=0.5, b2=1.0):
    """
    Analytic decomposition of the total effect of T on E[Y] for the
    chain SEM, returned as a dict keyed by path name.
    """
    return {
        "direct": b1,
        "chain_t_m1_m2_y": a1 * a2 * b2,
        "total": b1 + a1 * a2 * b2,
    }


# ============================================================
# Experiment 2 — Parallel mediators with interaction
#
#   T  ~ Bernoulli(0.5)
#   M1 = alpha1 * T + U1
#   M2 = alpha2 * T + U2
#   Y  = beta1*T + beta2*M1 + beta3*M2 + beta4*M1*T + beta5*M1*M2 + UY
#
# Columns: [T, M1, M2]. No confounder.
# Path T -> M1 -> Y ground truth: alpha1 * (beta2 + beta4 + alpha2*beta5).
# Path T -> M2 -> Y ground truth: alpha2 * (beta3 + alpha1*beta5).
# ============================================================

PAR_T_COL = 0
PAR_M1_COL = 1
PAR_M2_COL = 2


def parallel_interaction_dataset(n=5000,
                                  alpha1=0.7, alpha2=0.5,
                                  beta1=0.3, beta2=0.6, beta3=0.4,
                                  beta4=0.2, beta5=0.0,
                                  sigma=0.5, seed=0):
    rng = np.random.default_rng(seed)
    T = rng.binomial(1, 0.5, size=n).astype(float)
    U1 = rng.normal(0.0, sigma, size=n)
    U2 = rng.normal(0.0, sigma, size=n)
    UY = rng.normal(0.0, sigma, size=n)
    M1 = alpha1 * T + U1
    M2 = alpha2 * T + U2
    Y = (beta1 * T + beta2 * M1 + beta3 * M2
         + beta4 * M1 * T + beta5 * M1 * M2 + UY)
    X = np.column_stack([T, M1, M2])
    return X, Y


class ParallelModelWrapper:
    def __init__(self, beta1=0.3, beta2=0.6, beta3=0.4, beta4=0.2, beta5=0.0):
        self.b1 = beta1
        self.b2 = beta2
        self.b3 = beta3
        self.b4 = beta4
        self.b5 = beta5

    def predict(self, X):
        X = np.asarray(X)
        T = X[:, PAR_T_COL]
        M1 = X[:, PAR_M1_COL]
        M2 = X[:, PAR_M2_COL]
        return (self.b1 * T + self.b2 * M1 + self.b3 * M2
                + self.b4 * M1 * T + self.b5 * M1 * M2)


def parallel_ground_truth(alpha1=0.7, alpha2=0.5,
                           beta1=0.3, beta2=0.6, beta3=0.4,
                           beta4=0.2, beta5=0.0):
    """
    Analytic path effects for the parallel SEM (mean over the noise
    distribution, evaluated at the data-generating Bernoulli treatment).
    """
    path_m1 = alpha1 * (beta2 + beta4 + alpha2 * beta5)
    path_m2 = alpha2 * (beta3 + alpha1 * beta5)
    # Direct effect: d/dT of (beta1*T + beta4*M1*T) with M1 held at its
    # conditional mean alpha1*T under T=1 (cf. motivational example).
    direct = beta1
    # Interaction surplus that causes raw path-sum double counting.
    interaction_surplus = alpha1 * alpha2 * beta5
    return {
        "direct": direct,
        "path_t_m1_y": path_m1,
        "path_t_m2_y": path_m2,
        "interaction_surplus": interaction_surplus,
        "total": direct + path_m1 + path_m2 - interaction_surplus,
    }


# ============================================================
# Experiment 3 — Combined realistic DAG
#
#   C  ~ N(0.3, 0.5^2)
#   T  ~ Bernoulli(clip(0.8 - C, 0, 1))
#   M1 = 0.5*T + eps1
#   M2 = 0.7*M1 + 0.3*T + eps2
#   Y  = C + T - 0.8*M1^2 - 0.5*T*M1 + 0.4*M1*M2 + eps_Y
#
# Contains: confounder C, chain path (T->M1->M2->Y via M1*M2),
# direct path (T->Y), parallel path (T->M1->Y via M1^2 and T*M1),
# parallel path (T->M2->Y via M1*M2 when T->M2 arrow is also active),
# and nonlinearity via M1^2.
# ============================================================

COMB_T_COL = 0
COMB_C_COL = 1
COMB_M1_COL = 2
COMB_M2_COL = 3


def combined_realistic_dataset(n=5000, sigma=0.5, seed=0):
    rng = np.random.default_rng(seed)
    C = rng.normal(0.3, 0.5, size=n)
    p_t = np.clip(0.8 - C, 0.0, 1.0)
    T = rng.binomial(1, p_t).astype(float)
    eps1 = rng.normal(0.0, sigma, size=n)
    eps2 = rng.normal(0.0, sigma, size=n)
    epsY = rng.normal(0.0, sigma, size=n)
    M1 = 0.5 * T + eps1
    M2 = 0.7 * M1 + 0.3 * T + eps2
    Y = C + T - 0.8 * M1 ** 2 - 0.5 * T * M1 + 0.4 * M1 * M2 + epsY
    X = np.column_stack([T, C, M1, M2])
    return X, Y


class CombinedModelWrapper:
    def predict(self, X):
        X = np.asarray(X)
        T = X[:, COMB_T_COL]
        C = X[:, COMB_C_COL]
        M1 = X[:, COMB_M1_COL]
        M2 = X[:, COMB_M2_COL]
        return C + T - 0.8 * M1 ** 2 - 0.5 * T * M1 + 0.4 * M1 * M2


# ============================================================
# Canonical causal structures (C1-C6) for Shapley-variants
# comparison in Section "Comparing Shapley Variants"
# (experiments_and_results, Table tab:shapley-comparison).
# Each generator follows the SCM and feature layout shown in
# Figure fig:shapley-structures.
# ============================================================


def c1_chain(num_samples=1000, seed=0):
    """C1 - Chain: X1 -> X2, Y = X2.

    Features: [X1, X2]. Sample point in chapter: (x1=1, x2=2).
    """
    rng = np.random.default_rng(seed)
    X1 = rng.normal(0.5, 0.5, size=num_samples)
    X2 = X1 + rng.normal(0.5, 0.5, size=num_samples)
    Y = X2
    features = np.column_stack([X1, X2])
    return features, Y


def c2_fork(num_samples=1000, seed=0):
    """C2 - Fork: X2 -> X1, X2 -> Y.

    Features: [X1, X2]. Sample point in chapter: (x1=2.5, x2=1.5).
    """
    rng = np.random.default_rng(seed)
    X2 = rng.normal(0.5, 0.5, size=num_samples)
    X1 = X2 + rng.normal(0.5, 0.5, size=num_samples)
    Y = X2
    features = np.column_stack([X1, X2])
    return features, Y


def c3_unobserved_feature_confounder(num_samples=1000, seed=0):
    """C3 - Unobserved feature confounder: X3 -> X1, X3 -> X2, X2 -> Y. X3 hidden.

    Features: [X1, X2] (X3 not in features). Sample: (x1=2, x2=2).
    """
    rng = np.random.default_rng(seed)
    X3 = rng.normal(0.5, 0.5, size=num_samples)
    X1 = X3 + rng.normal(0.5, 0.5, size=num_samples)
    X2 = X3 + rng.normal(0.5, 0.5, size=num_samples)
    Y = X2
    features = np.column_stack([X1, X2])
    return features, Y


def c4_observed_feature_confounder(num_samples=1000, seed=0):
    """C4 - Observed feature confounder: X3 -> X1, X3 -> X2, X2 -> Y. X3 in features.

    Features: [X1, X2, X3]. Sample: (x1=2, x2=2, x3=1).
    """
    rng = np.random.default_rng(seed)
    X3 = rng.normal(0.5, 0.5, size=num_samples)
    X1 = X3 + rng.normal(0.5, 0.5, size=num_samples)
    X2 = X3 + rng.normal(0.5, 0.5, size=num_samples)
    Y = X2
    features = np.column_stack([X1, X2, X3])
    return features, Y


def c5_collider(num_samples=1000, seed=0):
    """C5 - Collider: X1 -> X3 <- X2, X3 -> Y, X2 -> Y.

    Features: [X1, X2, X3]. Sample: (x1=0.75, x2=0.75, x3=2.25).
    """
    rng = np.random.default_rng(seed)
    X1 = rng.normal(0.5, 0.5, size=num_samples)
    X2 = rng.normal(0.5, 0.5, size=num_samples)
    X3 = X1 + X2 + rng.normal(0.5, 0.5, size=num_samples)
    Y = X2 + X3
    features = np.column_stack([X1, X2, X3])
    return features, Y


def c6_unobserved_prediction_confounder(num_samples=1000, seed=0):
    """C6 - Unobserved prediction confounder: X3 -> X1, X3 -> Y, X2 -> Y. X3 hidden.

    Features: [X1, X2] (X3 not in features). Sample: (x1=2, x2=1).
    """
    rng = np.random.default_rng(seed)
    X3 = rng.normal(0.5, 0.5, size=num_samples)
    X1 = X3 + rng.normal(0.5, 0.5, size=num_samples)
    X2 = rng.normal(0.5, 0.5, size=num_samples)
    Y = X3 + X2
    features = np.column_stack([X1, X2])
    return features, Y
