from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def compute_path_wise_effects_reg(sample, custom_model, X_test1, treatment_col_index, mediator1_col_index, mediator2_col_index, confounder_col_index, model_type="mlp", compute_all=False):
    # Predict initial outcomes using the custom model (usually on full X_test1)
    y_hat_model = custom_model.predict(X_test1)
    T_test1 = X_test1[:, treatment_col_index]

    def train_inner_models(X, y, T, model_type="xgb"):
        """
        Train T-learner models using treated and control splits,
        excluding treatment column from features.
        """
        # Remove treatment column from features to avoid leakage
        X_no_treatment = np.delete(X, treatment_col_index, axis=1)

        X_treated = X_no_treatment[T == 1]
        y_treated = y[T == 1]
        X_control = X_no_treatment[T == 0]
        y_control = y[T == 0]

        if model_type == "xgb":
            model_mu1 = xgb.XGBRegressor()
            model_mu1.fit(X_treated, y_treated)
            model_mu0 = xgb.XGBRegressor()
            model_mu0.fit(X_control, y_control)

        elif model_type == "mlp":
            model_mu1 = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
            model_mu1.fit(X_treated, y_treated)
            model_mu0 = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
            model_mu0.fit(X_control, y_control)

        elif model_type == "linear":
            model_mu1 = LinearRegression()
            model_mu1.fit(X_treated, y_treated)
            model_mu0 = LinearRegression()
            model_mu0.fit(X_control, y_control)

        elif model_type == "poly2":
            poly = PolynomialFeatures(degree=2, include_bias=False)
            model_mu1 = make_pipeline(poly, LinearRegression())
            model_mu1.fit(X_treated, y_treated)
            model_mu0 = make_pipeline(poly, LinearRegression())
            model_mu0.fit(X_control, y_control)

        else:
            raise ValueError(f"Invalid model_type {model_type}. Choose from 'xgb', 'mlp', 'linear', or 'poly2'.")

        return model_mu1, model_mu0

    def predict_cate(model_mu1, model_mu0, X):
        # Remove treatment column before prediction
        T = X[:, treatment_col_index]
        X_1 = X[T == 1]
        X_0 = X[T == 0]

        X_0=np.delete(X_0, treatment_col_index, axis=1)
        X_1=np.delete(X_1, treatment_col_index, axis=1)

        mu1 = model_mu1.predict(X_1)
        mu0 = model_mu0.predict(X_0)
        cate_mean = np.mean(mu1) - np.mean(mu0)
        return mu1, mu0, cate_mean

    # === Train models on full dataset ===
    def filter_close(X, col_idx, val, tol=0.1):
        return np.abs(X[:, col_idx] - val) < tol
    model_mu1, model_mu0 = train_inner_models(X_test1, y_hat_model, T_test1, model_type)

    mask_conf = filter_close(X_test1, confounder_col_index, sample[confounder_col_index])
    X_filtered_conf = X_test1[mask_conf]
    # === 1. Full data (all mediators present) ===
    mu1_full, mu0_full, cate_all = predict_cate(model_mu1, model_mu0, X_filtered_conf)

    # Helper function for filtering samples close to sample value

    # === 2. Remove Mediator 1: filter by mediator2 + confounder only ===
    mask_m2 = filter_close(X_test1, mediator2_col_index, sample[mediator2_col_index]) & \
              filter_close(X_test1, confounder_col_index, sample[confounder_col_index])
    X_filtered_m2 = X_test1[mask_m2]
    mu1_m2, mu0_m2, cate_mediator2 = predict_cate(model_mu1, model_mu0, X_filtered_m2)

    # === 3. Remove Mediator 2: filter by mediator1 + confounder only ===
    mask_m1 = filter_close(X_test1, mediator1_col_index, sample[mediator1_col_index]) & \
              filter_close(X_test1, confounder_col_index, sample[confounder_col_index])
    X_filtered_m1 = X_test1[mask_m1]
    mu1_m1, mu0_m1, cate_mediator1 = predict_cate(model_mu1, model_mu0, X_filtered_m1)

    # === 4. Remove Both Mediators: filter by confounder only ===
    mask_m1_m2 = filter_close(X_test1, mediator1_col_index, sample[mediator1_col_index]) & \
              filter_close(X_test1, confounder_col_index, sample[confounder_col_index]) & \
              filter_close(X_test1, mediator2_col_index, sample[mediator2_col_index])

    X_filtered_m1_m2 = X_test1[mask_m1_m2]
    mu1_conf, mu0_conf, cate_mediator1_mediator2 = predict_cate(model_mu1, model_mu0, X_filtered_m1_m2)

    # === Compute Path-wise SHAP values (effects) ===
    path_wise_shap_t_m2_y = cate_all - cate_mediator1
    path_wise_shap_t_m1_y = cate_all - cate_mediator2
    path_wise_shap_t_m1_m2_y = cate_all - cate_mediator1_mediator2

    pishap_t_m1_y = cate_mediator1_mediator2 - cate_mediator1
    pishap_t_m2_y = cate_mediator1_mediator2 - cate_mediator2
    pishap_t_m1_m2_y = cate_mediator1_mediator2 - cate_all
    pishap_t_y = cate_all

    return {
        "path_wise_shap_t_m2_y": path_wise_shap_t_m2_y,
        "path_wise_shap_t_m1_y": path_wise_shap_t_m1_y,
        "pishap_t_m1_y": pishap_t_m1_y,
        "pishap_t_m2_y": pishap_t_m2_y,
        "pishap_t_m1_m2_y": pishap_t_m1_m2_y,
        "pishap_t_y": pishap_t_y,
        "path_wise_shap_t_m1_m2_y": path_wise_shap_t_m1_m2_y,
        "cate_all": cate_all,
        "cate_but_mediator1": cate_mediator2,
        "cate_but_mediator2": cate_mediator1,
        "cate_mediator1_mediator2": cate_mediator1_mediator2
    } 
    
def compute_path_wise_effects_doubly_robust(sample, custom_model, X_test1, treatment_col_index, mediator1_col_index, mediator2_col_index, confounder_col_index, model_type="mlp", propensity_model_type="logistic"):
    # Predict using the outer model
    y_hat_model = custom_model.predict(X_test1)
    T_test1 = X_test1[:, treatment_col_index]  # Treatment column

    # === Compute Propensity Scores ===
    X_covariates = X_test1[:, confounder_col_index].reshape(-1, 1)
    if propensity_model_type == "logistic":    
        propensity_model = LogisticRegression()
    elif propensity_model_type=="poly2":
        propensity_model = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LogisticRegression())
    elif propensity_model_type=="mlp":
        propensity_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    elif propensity_model_type=="xgb":
        propensity_model = xgb.XGBClassifier()
    elif propensity_model_type=="logistic_invalid":
        pass
    else:
        raise ValueError("Invalid propensity_model_type. Choose from 'logistic' or 'poly2' or 'mlp' or 'xgb' or 'logistic_invalid'.")   

    if propensity_model_type != "logistic_invalid":
        propensity_model.fit(X_covariates, T_test1)
        propensity_scores = propensity_model.predict_proba(X_covariates)[:, 1]
    # Compute IPW weights
    else:
        propensity_scores = np.random.rand(len(X_covariates))

    weights = np.where(
        T_test1 == 1,
        1 / propensity_scores,
        1 / (1 - propensity_scores)
    )


    # Predict CATE for a given sample input.
    # We use a small epsilon to avoid floating point errors.
    # We use a mask to avoid computing the CATE for all samples.

    def train_inner_models(X, y, T, model_type="xgb"):
        """
        Train T-learner models using treated and control splits.
        
        Parameters:
        - X: features
        - y: outcomes
        - T: treatment indicator (0/1)
        - model_type: "xgb", "mlp", or "linear" (default "xgb")
        
        Returns:
        - model_mu1, model_mu0: trained models for treated and control groups
        """
        X_treated, y_treated = X[T == 1], y[T == 1]
        X_control, y_control = X[T == 0], y[T == 0]

        if model_type == "xgb":
            model_mu1 = xgb.XGBRegressor()
            model_mu1.fit(X_treated, y_treated)
            model_mu0 = xgb.XGBRegressor()
            model_mu0.fit(X_control, y_control)

        elif model_type == "mlp":
            model_mu1 = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
            model_mu1.fit(X_treated, y_treated)
            model_mu0 = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
            model_mu0.fit(X_control, y_control)

        elif model_type == "linear":
            model_mu1 = LinearRegression()
            model_mu1.fit(X_treated, y_treated)
            model_mu0 = LinearRegression()
            model_mu0.fit(X_control, y_control)
        
        elif model_type == "poly2":
            poly = PolynomialFeatures(degree=2, include_bias=False)
            model_mu1 = make_pipeline(poly, LinearRegression())
            model_mu1.fit(X_treated, y_treated)
            model_mu0 = make_pipeline(poly, LinearRegression())
            model_mu0.fit(X_control, y_control)
        else:
            raise ValueError("Invalid model_type. Choose from 'xgb', 'mlp', or 'linear'.")

        return model_mu1, model_mu0


    def predict_cate(model_mu1, model_mu0, X_subset, y_subset, propensity_scores_subset):
        from numpy import nan
        T_sub = X_subset[:, treatment_col_index]
        Y_sub = y_subset
        e_sub = propensity_scores_subset
        X_sub = X_subset

        # Predict mu1 and mu0 for each X in the neighborhood
        X_sub_no_treatment = np.delete(X_sub, treatment_col_index, axis=1)

        X_sub_1 = X_sub_no_treatment[T_sub == 1]
        X_sub_0 = X_sub_no_treatment[T_sub == 0]


        Y_sub_1 = Y_sub[T_sub == 1]
        Y_sub_0 = Y_sub[T_sub == 0]

        if len(np.array(X_sub_1)) == 0 or len(np.array(X_sub_0)) == 0:
            return nan
        mu1_pred = model_mu1.predict(X_sub_1)
        mu0_pred = model_mu0.predict(X_sub_0)

        e_sub_1 = e_sub[T_sub == 1]
        e_sub_0 = e_sub[T_sub == 0]

        # # Doubly robust estimates
        reg_mu1 = np.mean(mu1_pred)
        reg_mu0 = np.mean(mu0_pred)
        ipw_mu1 = np.sum((Y_sub_1 - mu1_pred) / e_sub_1) / np.sum(1 / e_sub_1)
        ipw_mu0 = np.sum((Y_sub_0 - mu0_pred) / (1 - e_sub_0)) / np.sum(1 / (1 - e_sub_0))
        # print("REG")
        # print(mu1_pred)
        # print("Correction")
        # print((Y_sub_1 - mu1_pred) / e_sub_1)
        return  ipw_mu1 - ipw_mu0 + reg_mu1 - reg_mu0
    
    # === 1. Full model: no mediators removed ===
    X_S = np.delete(X_test1, treatment_col_index, axis=1)
    
    models = train_inner_models(X_S, y_hat_model, T_test1, model_type)

    mask_conf = np.abs(X_test1[:, confounder_col_index] - sample[confounder_col_index]) <= 0.1
    X_filtered_conf = X_test1[mask_conf]

    cate_confounder = predict_cate(*models, X_filtered_conf, y_hat_model[mask_conf], propensity_scores[mask_conf])
    # === 2. Remove Mediator 1: filter by mediator2 + confounders ===
    mask_m2 = (np.abs(X_test1[:, mediator2_col_index] - sample[mediator2_col_index]) <= 0.1) &\
                (np.abs(X_test1[:, confounder_col_index] - sample[confounder_col_index]) <= 0.1)

    X_filtered_m2 = X_test1[mask_m2]
    cate_mediator2_confounder = predict_cate(*models, X_filtered_m2, y_hat_model[mask_m2], propensity_scores[mask_m2])
    # === 3. Remove Mediator 2: filter by mediator1 + confounders ===
    mask_m1 = (np.abs(X_test1[:, mediator1_col_index] - sample[mediator1_col_index]) <= 0.1) &\
              (np.abs(X_test1[:, confounder_col_index] - sample[confounder_col_index]) <= 0.1)
    X_filtered_m1 = X_test1[mask_m1]
    cate_mediator1_confounder = predict_cate(*models, X_filtered_m1, y_hat_model[mask_m1], propensity_scores[mask_m1])

    # === 4. Remove Both Mediators: filter by confounders only ===
    mask_m1_m2 = (np.abs(X_test1[:, confounder_col_index] - sample[confounder_col_index]) <= 0.1) & \
                 (np.abs(X_test1[:, mediator1_col_index] - sample[mediator1_col_index]) <= 0.1) & \
                 (np.abs(X_test1[:, mediator2_col_index] - sample[mediator2_col_index]) <= 0.1)
    X_filtered_m1m2 = X_test1[mask_m1_m2]

    cate_mediator1_mediator2_confounder = predict_cate(*models, X_filtered_m1m2, y_hat_model[mask_m1_m2], propensity_scores[mask_m1_m2])

    # === Compute SHAP values ===
    path_wise_shap_t_m2_y = cate_mediator1_mediator2_confounder - cate_mediator1_confounder
    path_wise_shap_t_m1_y = cate_mediator1_mediator2_confounder - cate_mediator2_confounder
    path_wise_shap_t_m1_m2_y = cate_mediator1_mediator2_confounder - cate_confounder

    pishap_t_m1_y = cate_confounder - cate_mediator1_confounder
    pishap_t_m2_y = cate_confounder - cate_mediator2_confounder
    pishap_t_m1_m2_y = cate_confounder - cate_mediator1_mediator2_confounder
    pishap_t_y = cate_mediator1_mediator2_confounder

    return {
        "path_wise_shap_t_m2_y": path_wise_shap_t_m2_y,
        "path_wise_shap_t_m1_y": path_wise_shap_t_m1_y,
        "pishap_t_m1_y": pishap_t_m1_y,
        "pishap_t_m2_y": pishap_t_m2_y,
        "pishap_t_m1_m2_y": pishap_t_m1_m2_y,
        "pishap_t_y": pishap_t_y,
        "path_wise_shap_t_m1_m2_y": path_wise_shap_t_m1_m2_y,
        "cate_mediator1_mediator2_confounder": cate_mediator1_mediator2_confounder,
        "cate_mediator1_confounder": cate_mediator1_confounder,
        "cate_mediator2_confounder": cate_mediator2_confounder,
        "cate_confounder": cate_confounder
    }


def compute_path_wise_effects_ipw(sample, custom_model, X_test1, treatment_col_index, mediator1_col_index, mediator2_col_index, confounder_col_index, propensity_model_type="logistic"):
    # Predict using the outer model
    y_hat_model = custom_model.predict(X_test1)
    T_test1 = X_test1[:, treatment_col_index]  # Treatment column

    # === Compute Propensity Scores ===
    X_covariates = X_test1[:, confounder_col_index].reshape(-1, 1)

    if propensity_model_type == "logistic":    
        propensity_model = LogisticRegression()
    elif propensity_model_type=="poly2":
        propensity_model = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LogisticRegression())
    elif propensity_model_type=="mlp":
        propensity_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    elif propensity_model_type=="xgb":
        propensity_model = xgb.XGBClassifier()
    elif propensity_model_type=="logistic_invalid":
        pass
    else:
        raise ValueError("Invalid propensity_model_type. Choose from 'logistic' or 'poly2' or 'mlp' or 'xgb' or 'logistic_invalid'.")   

    if propensity_model_type != "logistic_invalid":
        propensity_model.fit(X_covariates, T_test1)
        propensity_scores = propensity_model.predict_proba(X_covariates)[:, 1]
    # Compute IPW weights
    else:
        propensity_scores = np.random.rand(len(X_covariates))

    weights = np.where(
        T_test1 == 1,
        1 / propensity_scores,
        1 / (1 - propensity_scores)
    )

    # Predict CATE for a given sample input.
    # We use a small epsilon to avoid floating point errors.
    # We use a mask to avoid computing the CATE for all samples.
    def predict_cate(X_S, sample_input, epsilon=0.2):
        mask = np.all(np.abs(X_S - sample_input) < epsilon, axis=1)

        if np.sum(mask) == 0:
            raise ValueError("No samples match the conditioning input.")
        
        treated = (T_test1 == 1) & mask
        control = (T_test1 == 0) & mask

        y_treated, w_treated = y_hat_model[treated], weights[treated]
        y_control, w_control = y_hat_model[control], weights[control]

        mu1 = np.sum(y_treated * w_treated) / np.sum(w_treated)
        mu0 = np.sum(y_control * w_control) / np.sum(w_control)

        return mu1 - mu0


    # === 1. Full model: no mediators removed ===
    X_S = np.delete(X_test1, [treatment_col_index, confounder_col_index], axis=1)
    sample_input = np.delete(np.array([sample]), [treatment_col_index, confounder_col_index], axis=1)

    cate_all = predict_cate(
        X_S,
        sample_input,
    )

    # === 2. Remove Mediator 1 ===
    X_S = np.delete(X_test1, obj=[treatment_col_index, confounder_col_index, mediator1_col_index], axis=1)
    sample_input = np.delete(np.array([sample]), obj=[treatment_col_index, confounder_col_index, mediator1_col_index], axis=1)

    cate_but_mediator1 = predict_cate(
        X_S,
        sample_input,

    )

    # === 3. Remove Mediator 2 ===
    X_S = np.delete(X_test1, obj=[treatment_col_index, confounder_col_index, mediator2_col_index], axis=1)
    sample_input = np.delete(np.array([sample]), obj=[treatment_col_index, confounder_col_index, mediator2_col_index], axis=1)

    cate_but_mediator2 = predict_cate(
        X_S,
        sample_input
    )

    # === 4. Remove Both Mediators ===
    X_S = np.delete(X_test1, obj=[treatment_col_index, mediator1_col_index, confounder_col_index, mediator2_col_index], axis=1)
    sample_input = np.delete(np.array([sample]), obj=[treatment_col_index, mediator1_col_index,confounder_col_index, mediator2_col_index], axis=1)

    cate_but_mediator1_mediator2 = predict_cate(
        X_S,
        sample_input
    )

    # === Compute SHAP values ===
    path_wise_shap_t_m2_y = cate_all - cate_but_mediator2
    path_wise_shap_t_m1_y = cate_all - cate_but_mediator1
    path_wise_shap_t_m1_m2_y = cate_all - cate_but_mediator1_mediator2

    pishap_t_m1_y = cate_but_mediator1_mediator2 - cate_but_mediator2
    pishap_t_m2_y = cate_but_mediator1_mediator2 - cate_but_mediator1
    pishap_t_m1_m2_y = cate_but_mediator1_mediator2 - cate_all
    pishap_t_y = cate_all

    return {
        "path_wise_shap_t_m2_y": path_wise_shap_t_m2_y,
        "path_wise_shap_t_m1_y": path_wise_shap_t_m1_y,
        "pishap_t_m1_y": pishap_t_m1_y,
        "pishap_t_m2_y": pishap_t_m2_y,
        "pishap_t_m1_m2_y": pishap_t_m1_m2_y,
        "pishap_t_y": pishap_t_y,
        "path_wise_shap_t_m1_m2_y": path_wise_shap_t_m1_m2_y,
        "cate_all": cate_all,
        "cate_but_mediator1": cate_but_mediator1,
        "cate_but_mediator2": cate_but_mediator2,
        "cate_but_mediator1_mediator2": cate_but_mediator1_mediator2
    }

def compute_path_wise_effects_conditioning(sample, custom_model, X_test1, treatment_col_index, mediator1_col_index, mediator2_col_index, confounder_col_index, propensity_model_type="logistic"):
    # Predict using the outer model
    y_hat_model = custom_model.predict(X_test1)
    T_test1 = X_test1[:, treatment_col_index]  # Treatment column

    # === Compute Propensity Scores ===
    X_covariates = X_test1[:, confounder_col_index].reshape(-1, 1)

    if propensity_model_type == "logistic":    
        propensity_model = LogisticRegression()
    elif propensity_model_type=="poly2":
        propensity_model = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LogisticRegression())
    elif propensity_model_type=="mlp":
        propensity_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    elif propensity_model_type=="xgb":
        propensity_model = xgb.XGBClassifier()
    elif propensity_model_type=="logistic_invalid":
        pass
    else:
        raise ValueError("Invalid propensity_model_type. Choose from 'logistic' or 'poly2' or 'mlp' or 'xgb' or 'logistic_invalid'.")   

    if propensity_model_type != "logistic_invalid":
        propensity_model.fit(X_covariates, T_test1)
        propensity_scores = propensity_model.predict_proba(X_covariates)[:, 1]
    # Compute IPW weights
    else:
        propensity_scores = np.random.rand(len(X_covariates))

    weights = np.where(
        T_test1 == 1,
        1 / propensity_scores,
        1 / (1 - propensity_scores)
    )

    # Predict CATE for a given sample input.
    # We use a small epsilon to avoid floating point errors.
    # We use a mask to avoid computing the CATE for all samples.
    def predict_cate(X_S, sample_input, epsilon=0.2):
        mask = np.all(np.abs(X_S - sample_input) < epsilon, axis=1)

        if np.sum(mask) == 0:
            raise ValueError("No samples match the conditioning input.")
        
        treated = (T_test1 == 1) & mask
        control = (T_test1 == 0) & mask

        y_treated = y_hat_model[treated]
        y_control = y_hat_model[control]

        mu1 = np.mean(y_treated)
        mu0 = np.mean(y_control)

        return mu1 - mu0


    # === 1. Full model: no mediators removed ===
    X_S = np.delete(X_test1, treatment_col_index, axis=1)
    sample_input = np.delete(np.array([sample]), treatment_col_index, axis=1)

    cate_all = predict_cate(
        X_S,
        sample_input,
    )

    # === 2. Remove Mediator 1 ===
    X_S = np.delete(X_test1, obj=[treatment_col_index, mediator1_col_index], axis=1)
    sample_input = np.delete(np.array([sample]), obj=[treatment_col_index, mediator1_col_index], axis=1)

    cate_but_mediator1 = predict_cate(
        X_S,
        sample_input,

    )

    # === 3. Remove Mediator 2 ===
    X_S = np.delete(X_test1, obj=[treatment_col_index, mediator2_col_index], axis=1)
    sample_input = np.delete(np.array([sample]), obj=[treatment_col_index, mediator2_col_index], axis=1)

    cate_but_mediator2 = predict_cate(
        X_S,
        sample_input
    )

    # === 4. Remove Both Mediators ===
    X_S = np.delete(X_test1, obj=[treatment_col_index, mediator1_col_index, mediator2_col_index], axis=1)
    sample_input = np.delete(np.array([sample]), obj=[treatment_col_index, mediator1_col_index, mediator2_col_index], axis=1)

    cate_but_mediator1_mediator2 = predict_cate(
        X_S,
        sample_input
    )

    # === Compute SHAP values ===
    path_wise_shap_t_m2_y = cate_all - cate_but_mediator2
    path_wise_shap_t_m1_y = cate_all - cate_but_mediator1
    path_wise_shap_t_m1_m2_y = cate_all - cate_but_mediator1_mediator2

    pishap_t_m1_y = cate_but_mediator1_mediator2 - cate_but_mediator2
    pishap_t_m2_y = cate_but_mediator1_mediator2 - cate_but_mediator1
    pishap_t_m1_m2_y = cate_but_mediator1_mediator2 - cate_all
    pishap_t_y = cate_all

    return {
        "path_wise_shap_t_m2_y": path_wise_shap_t_m2_y,
        "path_wise_shap_t_m1_y": path_wise_shap_t_m1_y,
        "pishap_t_m1_y": pishap_t_m1_y,
        "pishap_t_m2_y": pishap_t_m2_y,
        "pishap_t_m1_m2_y": pishap_t_m1_m2_y,
        "pishap_t_y": pishap_t_y,
        "path_wise_shap_t_m1_m2_y": path_wise_shap_t_m1_m2_y,
        "cate_all": cate_all,
        "cate_but_mediator1": cate_but_mediator1,
        "cate_but_mediator2": cate_but_mediator2,
        "cate_but_mediator1_mediator2": cate_but_mediator1_mediator2
    }




