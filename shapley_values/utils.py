from enum import Enum

def get_baseline(X, model, is_classification = False) -> float:
    """ Returns baseline value for computing ShapleY Values as averaged prediction across trainign set"""
    return model.predict(X).mean() if not is_classification else model.predict_proba(X).mean()

class ShapleyValuesType(Enum):
    MARGINAL = 'MARGINAL'
    CONDITIONAL = 'CONDITIONAL'
    CAUSAL = 'CAUSAL'
    PWCONDITIONAL = "PWCONDITIONAL"
