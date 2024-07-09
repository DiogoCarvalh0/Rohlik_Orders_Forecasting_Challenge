from pytorch_tabnet.metrics import Metric
from sklearn.metrics import mean_absolute_percentage_error


class MAPE(Metric):
    def __init__(self):
        self._name = "MAPE"
        self._maximize = False

    def __call__(self, y_true, y_pred):
        return mean_absolute_percentage_error(y_true, y_pred)
