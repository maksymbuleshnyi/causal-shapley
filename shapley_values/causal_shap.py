import itertools
import numpy as np
import random
import collections
from shapley_values.probabilities import conditional_prob, get_probability
from shapley_values.utils import get_baseline, ShapleyValuesType
from shapley_values.exceptions import CausalModelException
from pydantic import BaseModel
from typing import Any, List
import networkx as nx

random.seed(42)

class Explainer(BaseModel):
    X: Any
    model: Any
    feature_names: list[str]
    is_classification: bool = False
    X_counter: collections.Counter = collections.Counter()
    rounding_precision: int = 2
    propensity_models: dict[int, Any] = None

    def __init__(self, **data):
        super().__init__(**data)
        self.X = np.round(self.X, self.rounding_precision)
        self.X_counter = collections.Counter(map(tuple, self.X))

    def propensity_weight(self, xi_value, propensity_score):
        if xi_value == 1:
            return 1 / (propensity_score + 1e-9)
        else:
            return 1 / (1 - propensity_score + 1e-9)

    def validate_causal_model(self, causal_model: Any) -> dict[int, list[int]]:
        """
        Validates if passed Causal Model is correct. Returns causal model with name of feature changed on their index. 

        Raises:
            CausalModelException: If the provided causal model is incorrect.

        Returns:
            dict[int, list[int]]: Returns causal model with name of feature changed on their index.
        """

        if not isinstance(causal_model, dict):
            raise CausalModelException(f"Error: Causal model has to be provided as dictionary.")

        element_to_index = {element: index for index, element in enumerate(self.feature_names)}
        
        result = {}
        try:
            for key, value_list in causal_model.items():
                key_index = element_to_index[key]
                value_indices = [element_to_index[value] for value in value_list]
                result[key_index] = value_indices
        except Exception as e:
            raise CausalModelException("Error: Name of features were not recognized."
                                       f"Make sure you use the same feature names you provided to Explainer. {e}")

        G = nx.DiGraph(result)
        if not nx.is_directed_acyclic_graph(G):
            raise CausalModelException(f"Error: Causal model has cycles.")
        
        return result

    def compute_shapley_values(self, sample: List, type = ShapleyValuesType.MARGINAL, is_asymmetric = False, causal_model: dict = None, features_to_compute = []):
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
        f_x = get_baseline(self.X, self.model)
        if type == ShapleyValuesType.CAUSAL:
            if not causal_model:
                raise CausalModelException(
                    "Error: Causal graph has to be provided for computing Causal Shapley values")
            else:
                causal_model = self.validate_causal_model(causal_model)

        
        if is_asymmetric:
            if not causal_model:
                raise CausalModelException(
                    "Error: Causal graph has to be provided for computing Asymmetric Shapley values.")
            else:
                causal_model = self.validate_causal_model(causal_model)
                

        sample = np.round(sample, self.rounding_precision)
        n_features = self.X.shape[-1]
        phis = []
        for feature in range(n_features):

            if not features_to_compute or feature in features_to_compute:
                local_shap_score = self.approximate_shapley(feature, sample, type,
                                                            causal_model, is_asymmetric)
                
                phis.append(local_shap_score)
        
        # Check if the sum of the Shapley values and expected value adds up to the prediction
        x = np.reshape(sample, (1, n_features))
        f_x = get_baseline(self.X, self.model)

        print("Baseline Value (E[f(X)]): ", f_x)
        print("Predicted Value (f(x)) ", self.model.predict(x))
        print("Shapley Values + (E[f(X)]): ",
              round(float(sum(phis) + f_x), 3))

        return phis
    
    def get_all_parents(self, node, graph, visited=None):
        if visited is None:
            visited = set()

        if node in visited:
            return []

        visited.add(node)

        parents = []
        
        for parent, children in graph.items():
            if node in children:
                parents.append(parent)
                parents.extend(self.get_all_parents(parent, graph, visited))

        return list(set(parents))

    def follows_causal_structure(self, permutation: list[int], causal_struct: dict[int, list[int]]) -> bool:
        for index, feature in enumerate(permutation):
            parents = self.get_all_parents(feature, causal_struct)
            if not set(parents).issubset(permutation[:index]):
                return False

        return True

    def approximate_shapley(self, xi: int, x: list[int], type: ShapleyValuesType, causal_struct: dict[int, list[int]] = None, is_asymmetric = False):
        N = self.X.shape[-1]
        m = 0
        R = list(itertools.permutations(range(N)))
        random.shuffle(R)
        score = 0
        count_negative = 0
        vf1, vf2 = 0, 0
        for index, permutation in enumerate(R):

            if not is_asymmetric or (is_asymmetric and self.follows_causal_structure(permutation, causal_struct)):
                abs_diff, f1, f2 = self.get_value(type, list(permutation), x, causal_struct,
                                                xi)
                
                print(abs_diff)
                print([permutation])
                m += 1
                vf1 += f1
                vf2 += f2

                score += abs_diff

        if count_negative < 0:
            score = -1 * score
        
        return score / m


    def get_value(self, type, permutation, x, causal_struct, xi):
        N = self.X.shape[-1]

        lenX = self.X.shape[0]
        absolute_diff, f1, f2 = 0, 0, 0
        xi_index = permutation.index(xi)
        indices = permutation[:xi_index + 1]
        indices_baseline = permutation[xi_index + 1:]
        x_hat = np.zeros(N)
        x_hat_2 = np.zeros(N)
        f2_values = []
        propensity_weight_f1 = 0
        propensity_weight_f2 = 0

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
                match type:
                    case ShapleyValuesType.MARGINAL:
                        prob_x_hat = get_probability(
                            self.X_counter, x_hat, indices_baseline, lenX)
                    case ShapleyValuesType.CONDITIONAL:
                        prob_x_hat = conditional_prob(
                            self.X_counter, x_hat, indices, indices_baseline, lenX)
                    case ShapleyValuesType.CAUSAL:
                        prob_x_hat = 0.1  # Implementation with do intervetions
                    case ShapleyValuesType.PWCONDITIONAL:
                        prob_x_hat = conditional_prob(
                            self.X_counter, x_hat, indices, indices_baseline, lenX)

                x_hat = np.reshape(x_hat, (1, N))

                if type == ShapleyValuesType.PWCONDITIONAL:
                    treatment_value = x_hat[0, xi]
                    propensity_input = np.delete(x_hat, xi, axis=1)

                    if xi in self.propensity_models:
                        propensity_score = self.propensity_models[xi].predict_proba(propensity_input)[:, 1]
                        propensity_weight_f1 += self.propensity_weight(treatment_value, propensity_score)

                        f1 = f1 + self.propensity_weight(treatment_value, propensity_score) * (self.model.predict_proba(x_hat)[0] if self.is_classification else self.model.predict(
                            x_hat)[0])
                    else:
                        f1 = f1 + (self.model.predict_proba(x_hat)[0] if self.is_classification else self.model.predict(
                            x_hat)[0])

                else:
                    f1 = f1 + (self.model.predict_proba(x_hat)[0] * prob_x_hat if self.is_classification else self.model.predict(
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
                match type:
                    case ShapleyValuesType.MARGINAL:
                        prob_x_hat_2 = get_probability(
                            self.X_counter, x_hat_2, indices_baseline_2, lenX)
                    case ShapleyValuesType.CONDITIONAL:
                        prob_x_hat_2 = conditional_prob(
                            self.X_counter, x_hat_2, indices_2, indices_baseline_2, lenX)
                    case ShapleyValuesType.CAUSAL:
                        prob_x_hat_2 = 0.1 # Implementation with do intervetions
                    
                    case ShapleyValuesType.PWCONDITIONAL:
                        prob_x_hat_2 = conditional_prob(
                            self.X_counter, x_hat_2, indices_2, indices_baseline_2, lenX)

                x_hat_2 = np.reshape(x_hat_2, (1, N))
                
                if type == ShapleyValuesType.PWCONDITIONAL:
                    treatment_value = x_hat_2[0, xi]

                    propensity_input = np.delete(x_hat_2, xi, axis=1)

                    if xi in self.propensity_models:
                        propensity_score = self.propensity_models[xi].predict_proba(propensity_input)[:, 1]
                        propensity_weight_f2 += self.propensity_weight(treatment_value, propensity_score)
                        f2_values[x_hat_2] = self.propensity_weight(treatment_value, propensity_score) * (self.model.predict_proba(x_hat_2)[0][1] if self.is_classification else self.model.predict(
                            x_hat_2)[0])
                        f2 = f2 + self.propensity_weight(treatment_value, propensity_score) * (self.model.predict_proba(x_hat_2)[0][1] if self.is_classification else self.model.predict(
                            x_hat_2)[0])
                    else:
                        f2 = f2 + (self.model.predict_proba(x_hat_2)[0][1] * prob_x_hat_2 if self.is_classification else self.model.predict(
                            x_hat_2)[0])
                else:
                    f2 = f2 + (self.model.predict_proba(x_hat_2)[0][1] * prob_x_hat_2 if self.is_classification else self.model.predict(
                        x_hat_2)[0] * prob_x_hat_2)

            x_hat = np.squeeze(x_hat)
            x_hat_2 = np.squeeze(x_hat_2)

        if propensity_weight_f1 == 0:
            propensity_weight_f1 = 1
        
        if propensity_weight_f2 == 0:
            propensity_weight_f2 = 1

        absolute_diff = f1 / propensity_weight_f1 - f2 / propensity_weight_f2
        return absolute_diff, f1 / propensity_weight_f1, f2 / propensity_weight_f2


# Some of the code sample are taken from https://github.com/saifkhanali9/causal-shapley