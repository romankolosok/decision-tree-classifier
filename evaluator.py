import pandas as pd


class BinaryClassificationEvaluator:
    """
    Computes common binary classification metrics from true and predicted labels.

    Args:
        y_true:          Ground-truth class labels.
        y_pred:          Predicted class labels.
        positive_class:  The value treated as the positive class.
                         Defaults to the first unique value found in y_true.
    """

    def __init__(self, y_true: pd.Series, y_pred: pd.Series, positive_class=None):
        self.y_true = pd.Series(y_true).reset_index(drop=True)
        self.y_pred = pd.Series(y_pred).reset_index(drop=True)

        if positive_class is None:
            positive_class = self.y_true.unique()[0]
        self.positive_class = positive_class

        # Compute confusion matrix counts once, reused by all metrics
        self.tp = ((self.y_true == positive_class) & (self.y_pred == positive_class)).sum()
        self.fp = ((self.y_true != positive_class) & (self.y_pred == positive_class)).sum()
        self.fn = ((self.y_true == positive_class) & (self.y_pred != positive_class)).sum()
        self.tn = ((self.y_true != positive_class) & (self.y_pred != positive_class)).sum()

    def accuracy(self) -> float:
        return (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn)

    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    def false_positive_rate(self) -> float:
        return self.fp / (self.fp + self.tn) if (self.fp + self.tn) > 0 else 0.0

    def report(self) -> None:
        """Prints all metrics."""
        print(f"Accuracy:            {self.accuracy():.4f}")
        print(f"Precision:           {self.precision():.4f}")
        print(f"Recall:              {self.recall():.4f}")
        print(f"False-positive rate: {self.false_positive_rate():.4f}\n")