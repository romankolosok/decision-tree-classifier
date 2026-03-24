import math
from enum import Enum
from typing import List, Dict, Tuple, Optional

import pandas as pd


class ComparisonOperators(Enum):
    """Supported operators for splitting conditions."""
    LESS_THAN = "<"
    EQUALS = "="


class Condition:
    """
    Represents a single binary split condition of the form:
        feature[feature_index] <operator> value

    Used by the decision tree to partition data at each node.
    """

    def __init__(
            self,
            value: int,
            feature_index: int,
            operator: ComparisonOperators = ComparisonOperators.EQUALS,
    ):
        self.value = value
        self.feature_index = feature_index
        self.operator = operator

    def get_mask(self, data: pd.DataFrame) -> pd.Series:
        """
        Applies this condition to a DataFrame and returns a boolean mask.
        True means the row satisfies the condition (goes to the left/true branch).
        """
        if self.operator == ComparisonOperators.LESS_THAN:
            return data.iloc[:, self.feature_index] < self.value
        elif self.operator == ComparisonOperators.EQUALS:
            return data.iloc[:, self.feature_index] == self.value

        # Fallback: every row passes (should not be reached with valid operators)
        return pd.Series(True, index=data.index)

    def __gt__(self, other: "Condition") -> bool:
        """
        Ordering between two LESS_THAN conditions on the same feature.
        Used to prune redundant conditions when building child nodes.
        """
        if self.feature_index != other.feature_index:
            return False
        if self.operator == ComparisonOperators.LESS_THAN and other.operator == ComparisonOperators.LESS_THAN:
            return self.value > other.value
        return False

    def __lt__(self, other: "Condition") -> bool:
        """
        Ordering between two LESS_THAN conditions on the same feature.
        Used to prune redundant conditions when building child nodes.
        """
        if self.feature_index != other.feature_index:
            return False
        if self.operator == ComparisonOperators.LESS_THAN and other.operator == ComparisonOperators.LESS_THAN:
            return self.value < other.value
        return False

    def __repr__(self) -> str:
        return self.get_label()

    def get_label(self, column_names=None) -> str:
        """
        Returns a human-readable string for this condition.
        Uses the actual column name if `column_names` is provided,
        otherwise falls back to feature[i].
        """
        op = self.operator.value
        name = column_names[self.feature_index] if column_names is not None else f"feature[{self.feature_index}]"
        return f"{name} {op} {self.value}"


class NumericDataset:
    """
    Loads a CSV dataset and prepares it for use with DecisionTreeClassifier.

    - Reads the CSV and drops the first (index) column.
    - Performs a random train/test split.
    - Resolves the target column index (supports negative indexing).
    - Generates candidate split Conditions for all non-target features.
    """

    def __init__(
        self,
        filename: str,
        target: int = -1,
        test_proportion: float = 0.3,
        seed: int = 121,
    ):
        """
        Args:
            filename:         Path to the CSV file.
            target:           Column index of the target/label column.
                              Negative values are resolved relative to the end (e.g. -1 = last column).
            test_proportion:  Fraction of data held out for testing.
            seed:             Random seed for reproducible splits.
        """
        self.train = pd.read_csv(filename)
        self.train.drop(self.train.columns[0], axis=1, inplace=True)

        self.test = pd.DataFrame()
        self.num_features = len(self.train.columns)

        self.__train_test_split(test_proportion, seed)

        self.target = self.num_features + target if target < 0 else target

        # Precompute domains
        self.domains = [self.train.iloc[:, index].unique() for index in range(self.num_features)]

        self._conditions_cache: Dict[Tuple[int, tuple], List[Condition]] = {}

    def __train_test_split(self, test_proportion: float, seed: int) -> None:
        """
        Splits self.train into train and test subsets in-place.
        The test set is sampled without replacement, then removed from the training set.
        """
        self.test = self.train.sample(frac=test_proportion, replace=False, random_state=seed)
        self.train = self.train.drop(self.test.index)

    def get_conditions(self, max_cuts: int, categorical_cols: Optional[List[int]] = None) -> List[Condition]:
        """
        Generates and caches the list of candidate split Conditions.

        The strategy depends on the feature type:
        - Categorical feature (listed in categorical_cols):
            One EQUALS condition per unique value in the domain, regardless of domain size.
        - Binary non-categorical feature (2 unique values):
            A single EQUALS condition on the first value.
        - Continuous / multi-valued non-categorical feature:
            Up to `max_cuts` evenly-spaced LESS_THAN threshold conditions.
        """

        # Column indices of categorical features.
        if categorical_cols is None:
            categorical_cols = []

        # Resolve negative indices
        categorical_columns = [
            i if i >= 0 else i + self.num_features
            for i in categorical_cols
        ]

        cache_key = (max_cuts, tuple(sorted(categorical_columns)))
        if cache_key in self._conditions_cache:
            return self._conditions_cache[cache_key]

        conditions = []

        for column_index, domain in enumerate(self.domains):
            # Skip the target column and constant features (no split possible)
            if column_index == self.target or len(domain) <= 1:
                continue

            is_categorical = column_index in categorical_columns

            if is_categorical:
                for value in domain:
                    cond = Condition(
                        value=value,
                        feature_index=column_index,
                        operator=ComparisonOperators.EQUALS,
                    )
                    conditions.append(cond)

            elif len(domain) == 2:
                cond = Condition(
                    value=domain[0],
                    feature_index=column_index,
                    operator=ComparisonOperators.EQUALS,
                )
                conditions.append(cond)

            else:
                max_cut = min(max_cuts, len(domain))
                domain_sorted = sorted(domain)

                # Indices into domain_sorted, spaced to produce max_cut - 1 thresholds
                cuts = [(len(domain_sorted) // max_cut) * i for i in range(1, max_cut)]

                for domain_idx in cuts:
                    domain_cut = domain_sorted[domain_idx]
                    cond = Condition(
                        value=domain_cut,
                        feature_index=column_index,
                        operator=ComparisonOperators.LESS_THAN,
                    )
                    conditions.append(cond)

        self._conditions_cache[cache_key] = conditions
        return conditions