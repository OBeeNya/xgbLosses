import numpy as np

class XgbLosses:
    """
    Class containing custom objective functions for XGBoost.
    Implementing such a class allows to pass each function as hyperparameter of the model with different values for alpha, gamma...

    Parameters
    ----------
    pred : prediction vector
    dtrain : target vector, DMatrix
    
    Returns
    -------
    grad : vector of first order gradient based on model predictions
    hess : vector of second order gradient based on model predictions
    
    
    Reference : https://xgboost.readthedocs.io/en/stable/tutorials/custom_metric_obj.html
    """

    def __init__(self, alpha: float = 0.05, gamma: float = 3.0):
        """
        Assign values to parameters of cost functions when initializing the instance:
            - alpha : imbalance parameter for the weighted cross-entropy loss
            - gamma : added factor to the standard cross-entropy criterion for the focal loss
        """
        self.alpha = alpha
        self.gamma = gamma
        self.funcs = [
            'weighted_cross_entropy',
            'focal',
            'exponential',
            'log_cosh',
            'composite',
        ]

    def pow(self, base, power):
        """
        Robust power function for the focal loss computation.
        """
        return np.sign(base) * (np.abs(base)) ** (power)

    def weighted_cross_entropy(self, pred, dtrain):
        """
        Output predictions split by value 0.
        Apply sigmoid function to predictions.

        Reference : https://github.com/jhwjhw0123/Imbalance-XGBoost
        """
        label = dtrain.get_label()
        sigmoid_pred = 1.0 / (1.0 + np.exp(-pred))
        grad = - (self.alpha ** label) * (label - sigmoid_pred)
        hess = (self.alpha ** label) * sigmoid_pred * (1.0 - sigmoid_pred)
        return grad, hess
    
    def focal(self, pred, dtrain):
        """
        Output predictions split by value 0.
        Apply sigmoid function to predictions.

        Reference : https://github.com/jhwjhw0123/Imbalance-XGBoost
        """
        label = dtrain.get_label()
        sigmoid_pred = 1.0 / (1.0 + np.exp(-pred))
        g1 = sigmoid_pred * (1 - sigmoid_pred)
        g2 = label + ((-1) ** label) * sigmoid_pred
        g3 = sigmoid_pred + label - 1
        g4 = 1 - label - ((-1) ** label) * sigmoid_pred
        g5 = label + ((-1) ** label) * sigmoid_pred
        grad = self.gamma * g3 * self.pow(g2, self.gamma) * np.log(g4 + 1e-9) + ((-1) ** label) * self.pow(g5, (self.gamma + 1))
        hess_1 = self.pow(g2, self.gamma) + self.gamma * ((-1) ** label) * g3 * self.pow(g2, (self.gamma - 1))
        hess_2 = ((-1) ** label) * g3 * self.pow(g2, self.gamma) / g4
        hess = ((hess_1 * np.log(g4 + 1e-9) - hess_2) * self.gamma + (self.gamma + 1) * self.pow(g5, self.gamma)) * g1
        return grad, hess

    def exponential(self, pred, dtrain):
        """
        Output predictions split by value 0.5.
        """
        labels = dtrain.get_label() * 2.0 - 1.0
        grad = - np.exp(- pred * labels) * labels
        hess = np.exp(- pred * labels) * labels ** 2
        return grad, hess
    
    def log_cosh(self, pred, dtrain):
        """
        Output predictions split by value 0.5.
        """
        labels = dtrain.get_label()
        grad = np.tanh(pred - labels) / np.log(10)
        hess = ((1 / np.cosh(pred - labels)) ** 2) / np.log(10)
        return grad, hess

    def composite(self, pred, dtrain):
        """
        Output predictions split by value ???
        Not a good idea since output ranges are different for each objective function ?
        """
        grad1, hess1 = self.weighted_cross_entropy(pred, dtrain)
        grad2, hess2 = self.focal(pred, dtrain)
        grad3, hess3 = self.exponential(pred, dtrain)
        grad4, hess4 = self.log_cosh(pred, dtrain)
        grad = grad1 + grad2 + grad3 + grad4
        hess = hess1 + hess2 + hess3 + hess4
        return grad, hess
