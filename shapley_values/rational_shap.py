from typing import Any, Callable
from shapley_values123123123.utils import get_baseline, ShapleyValuesType
from shapley_values123123123.causal_shap import Explainer
from pydantic import BaseModel

class RationalExplainer(BaseModel):
    subspaces: list[list]
    model: Any
    feature_names: list[str]
    is_classification: bool = False
    rounding_precision: int = 2

    def compute_rational_shapley_values(self,
                                         sample,
                                         reward_function: Callable[[list[float]], float],
                                         causal_model: dict = None,
                                         type = ShapleyValuesType.MARGINAL) -> tuple[list[float], float]:
        """
        Computes rational shapley values.

        Args:
            sample (list): A list of feature values representing the input sample for which attributions are computed.
            credence_func (Callable): A function that computes the degree of belief in an outcome
                based on a given body of evidence (Shapley values). 
                It must accept two parameters: outcome and Shapley values.
            reward_function (Callable): A function that computes the reward for a user given Shapley values.
                It must accept two parameters: action and outcome.
            type (ShapleyValuesType): An enumeration value indicating the type of Shapley values to compute. 
                Options include:
                    - ShapleyValuesType.MARGINAL
                    - ShapleyValuesType.CONDITIONAL
                    - ShapleyValuesType.CAUSAL
            causal_struct (Dict): A dictionary defining the causal structure of the model, which is essential for causal calculations.
            
        """
        explainer = Explainer(X = self.subspaces[0],
                                   model = self.model,
                                   feature_names = self.feature_names,
                                   is_classification = self.is_classification,
                                     rounding_precision = self.rounding_precision)

        best_phis = explainer.compute_shapley_values(sample, causal_model=causal_model, type = type)
        max_reward = reward_function(best_phis)

        rewards_map = {max_reward: best_phis}

        for subspace in self.subspaces[1:]:
            explainer = Explainer(X = subspace,
                                   model = self.model,
                                   feature_names = self.feature_names,
                                   is_classification = self.is_classification,
                                     rounding_precision = self.rounding_precision)
            
            phis = explainer.compute_shapley_values(sample, causal_model=causal_model, type = type)
            reward = reward_function(phis)

            rewards_map[reward] = phis
            
        
        return rewards_map
