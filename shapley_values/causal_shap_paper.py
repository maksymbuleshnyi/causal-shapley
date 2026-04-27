
import itertools
import numpy as np
import random
import collections
from shapley_values.probabilities import causal_prob
from shapley_values.utils import get_baseline
from shapley_values.exceptions import CausalModelException
from shapley_values.causal_shap import Explainer
from typing import Any, List
import networkx as nx
from copy import deepcopy
random.seed(42)
from enum import Enum

class EffectType(Enum):
    TOTAL = 1
    DIRECT = 2
    INDIRECT = 3

class CausalExplainer(Explainer):
    def compute_shapley_values(self, sample: List, effect_type = EffectType.TOTAL, is_asymmetric = False, causal_model: list[list] = None, confounding: list = [], asymmetric_causal_model: dict = None, features_compute = []):
        """
        Computes the attribution of each feature for a given sample using Shapley values.

        Args:
            sample (list): A list of feature values representing the input sample for which attributions are computed.
            type (ShapleyValuesType): An enumeration value indicating the type of Shapley values to compute. 
                Options include:
                    - ShapleyValuesType.MARGINAL
                    - ShapleyValuesType.CONDITIONAL
                    - ShapleyValuesType.CAUSAL

            is_asymmetric (bool): Whether to compute Asymmetric Shapley values.
            causal_struct (Dict): A dictionary defining the causal structure of the model, which is essential for causal calculations.

        Prints:
            Baseline Value (E[f(X)]):
                The average predicted value across all training samples, serving as a reference point for attributions.
            Predicted Value (f(x)):
                The model's predicted output for the specified sample, reflecting the influence of the features.
            Shapley Values + Baseline Value:
                A combined metric that merges the calculated Shapley value attributions with the expected value of the predicted outcome, providing a comprehensive view of feature contributions.

        Raises:
            CausalGraphMissedException: If the provided causal graph is incorrect or incomplete, indicating a failure to compute attributions properly.

        Returns:
            list: An array containing the attribution values for each feature, ordered according to the input sample.
        """
        f_x = get_baseline(self.X, self.model, is_classification=self.is_classification)
        
        if not causal_model:
            raise CausalModelException(
                "Error: Causal graph has to be provided for computing Causal Shapley values")

        sample = np.round(sample, self.rounding_precision)
        n_features = self.X.shape[-1]
        phis = []
        for feature in range(n_features):
            if not features_compute or feature in features_compute:
                local_shap_score = self.approximate_shapley(feature, sample, effect_type, confounding,
                                                            causal_model, is_asymmetric, asymmetric_causal_model = asymmetric_causal_model)
                
                phis.append(local_shap_score)
        
        # Check if the sum of the Shapley values and expected value adds up to the prediction
        x = np.reshape(sample, (1, n_features))
        f_x = get_baseline(self.X, self.model)


        print("Baseline Value (E[f(X)]): ", f_x)
        print("Predicted Value (f(x)) ", self.model.predict_proba(x)[0] if self.is_classification else self.model.predict(x))
        print("Shapley Values + (E[f(X)]): ",
              round(float(sum(phis) + f_x), 3))

        return phis


    def approximate_shapley(self, xi: int, x: list[int], effect_type: EffectType, confounding, causal_struct: dict[int, list[int]] = None, is_asymmetric = False, asymmetric_causal_model: dict = None):
        N = self.X.shape[-1]
        m = 0
        R = list(itertools.permutations(range(N)))
        random.shuffle(R)
        score = 0
        vf1, vf2 = 0, 0
        for permutation in R:
            if not is_asymmetric or (is_asymmetric and self.follows_causal_structure(permutation, asymmetric_causal_model)):
                abs_diff, f1, f2 = self.get_value(effect_type, list(permutation), x, causal_struct,
                                                xi, confounding)
                m += 1
                vf1 += f1
                vf2 += f2
                score += abs_diff
        
        return score / m
        
    def get_value(self, effect_type, permutation, x, causal_struct, xi, confounding):
        N = self.X.shape[-1]
        lenX = self.X.shape[0]
        absolute_diff, f1, f2 = 0, 0, 0
        xi_index = permutation.index(xi)
        indices = permutation[:xi_index + 1]
        indices_baseline = permutation[xi_index + 1:]
        x_hat = np.zeros(N)
        x_hat_2 = np.zeros(N)

        for j in indices:
            x_hat[j] = x[j]
            x_hat_2[j] = x[j]

        baseline_check_1, baseline_check_2 = [], []
        f1, f2 = 0, 0
        indices_baseline_2 = indices_baseline[:]
        for i in self.X_counter:
            X = np.asarray(i)
            for j in indices_baseline:
                x_hat[j] = x_hat_2[j] = X[j]

            # No repetition
            # Eg if baseline_indices is null, it'll only run once as x_hat will stay the same over each iteration
            if x_hat.tolist() not in baseline_check_1:
                baseline_check_1.append(x_hat.tolist())
                match effect_type:
                    case EffectType.TOTAL:
                        prob_x_hat = causal_prob(
                            self.X_counter, x_hat, indices, indices_baseline, lenX, causal_struct=causal_struct, confounding = confounding)
                    case EffectType.DIRECT:
                        if xi in indices:
                            indices_new = deepcopy(indices)
                            indices_new.remove(xi)
                        else:
                            indices_new = indices

                        prob_x_hat = causal_prob(
                            self.X_counter, x_hat, indices_new, indices_baseline, lenX, causal_struct=causal_struct, confounding = confounding)

                    case EffectType.INDIRECT:
                        prob_x_hat = causal_prob(
                            self.X_counter, x_hat, indices, indices_baseline, lenX, causal_struct=causal_struct, confounding = confounding)

                x_hat = np.reshape(x_hat, (1, N))

                f1 = f1 + (self.model.predict_proba(x_hat)[0][0] * prob_x_hat if self.is_classification else self.model.predict(
                    x_hat)[0] * prob_x_hat)

            # xi index will be given to baseline for f2
            x_hat_2[xi] = X[xi]
            if xi not in indices_baseline_2:
                indices_baseline_2.append(xi)

            # No repetition
            indices_2 = indices[:]
            indices_2.remove(xi)
            if x_hat_2.tolist() not in baseline_check_2:
                baseline_check_2.append(x_hat_2.tolist())
                match effect_type:
                    case EffectType.TOTAL:
                        prob_x_hat_2 = causal_prob(
                            self.X_counter, x_hat_2, indices_2, indices_baseline_2, lenX, causal_struct=causal_struct, confounding = confounding)
                    
                    case EffectType.DIRECT:
                        if xi in indices_2:
                            indices_2.remove(xi)

                        prob_x_hat_2 = causal_prob(
                            self.X_counter, x_hat_2, indices_2, indices_baseline_2, lenX, causal_struct=causal_struct, confounding = confounding)
                    
                        
                    case EffectType.INDIRECT:
                        prob_x_hat_2 = causal_prob(
                            self.X_counter, x_hat_2, indices_2, indices_baseline_2, lenX, causal_struct=causal_struct, confounding = confounding)


                x_hat_2 = np.reshape(x_hat_2, (1, N))
                
                if effect_type == EffectType.INDIRECT:
                    x_hat = np.reshape(x_hat, (1, N))

                    f2 = f2 + (self.model.predict_proba(x_hat)[0][0] * prob_x_hat_2 if self.is_classification else self.model.predict(
                        x_hat)[0] * prob_x_hat_2)
                else:
                    f2 = f2 + (self.model.predict_proba(x_hat_2)[0][0] * prob_x_hat_2 if self.is_classification else self.model.predict(
                        x_hat_2)[0] * prob_x_hat_2)

            x_hat = np.squeeze(x_hat)
            x_hat_2 = np.squeeze(x_hat_2)
        absolute_diff = f1 - f2
        return absolute_diff, f1, f2
