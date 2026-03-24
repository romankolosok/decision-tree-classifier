from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from dataset import Condition, ComparisonOperators, NumericDataset


class Node:
    """
    A single node in the decision tree.

    Leaf nodes carry a `prediction` (majority class) and have no children.
    Internal nodes carry a `condition` that routes rows left (True) or right (False).
    """

    def __init__(
        self,
        prediction: int = None,
        left: Optional[Node] = None,
        right: Optional[Node] = None,
        condition: Optional[Condition] = None,
    ):
        self.prediction = prediction  # Set only on leaf nodes
        self.left = left              # Subtree for rows that satisfy `condition`
        self.right = right            # Subtree for rows that don't satisfy `condition`
        self.condition = condition    # Split rule; None on leaf nodes

    def is_leaf(self) -> bool:
        """Returns True if this node is a leaf (holds a prediction, no children)."""
        return self.prediction is not None


class DecisionTreeClassifier:
    """
    A binary decision tree classifier

    Splitting stops early when:
    - All remaining examples share the same label (pure node).
    - No condition yields an Information Gain above `gamma`.
    - The maximum tree depth `max_depth` is reached.

    Args:
        max_depth:        Maximum depth the tree is allowed to grow.
        max_cuts:         Maximum number of threshold cut points evaluated per feature.
        gamma:            Minimum Information Gain required to accept a split.
        categorical_cols: Column indices of categorical features passed through to
                          dataset.get_conditions(). Negative values are supported.
    """

    def __init__(
        self,
        max_depth: int = 8,
        max_cuts: int = 8,
        gamma: float = 10e-7,
        categorical_cols: Optional[List[int]] = None,
    ):
        self.max_depth = max_depth
        self.gamma = gamma
        self.max_cuts = max_cuts
        self.categorical_cols = categorical_cols or []
        self.root: Optional[Node] = None

    def fit(self, dataset: NumericDataset) -> None:
        """
        Builds the decision tree from the training split of `dataset`.
        Must be called before `predict` or `print_tree`.
        """
        if not isinstance(dataset, NumericDataset):
            raise ValueError("Input must be an instance of dataset.NumericDataset")

        conditions = dataset.get_conditions(max_cuts=self.max_cuts, categorical_cols=self.categorical_cols)
        self.root = self._build_tree(dataset.train, conditions, dataset.target, depth=0)

    def predict(self, dataset: NumericDataset, on_train: bool = False) -> pd.Series:
        """
        Predicts class labels for every row in the test set (or train set).

        Args:
            dataset:  The NumericDataset whose split is used for prediction.
            on_train: If True, predict on the training split instead of the test split.

        Returns:
            A pandas Series of predicted class labels aligned with the chosen split.
        """
        if self.root is None:
            raise Exception("Tree not trained. Call fit() first.")

        data = dataset.train if on_train else dataset.test

        # Drop the target column so we only pass features to the traversal
        X = data.drop(data.columns[dataset.target], axis=1)

        predictions = X.apply(lambda row: self._predict_one(row, self.root), axis=1)
        return predictions

    def print_tree(self, node: Optional[Node] = None, depth: int = 0, column_names=None) -> None:
        """
        Recursively prints the tree structure to stdout.

        Internal nodes show their split condition, leaf nodes show their prediction.
        """
        if node is None:
            if depth == 0:
                node = self.root
            else:
                return

        if node is None:
            print("Tree is not trained.")
            return

        indent = "  " * depth
        if node.is_leaf():
            print(f"{indent}Predict: {node.prediction}")
        else:
            condition_str = node.condition.get_label(column_names) if node.condition else "Condition"
            print(f"{indent}if {condition_str}:")
            self.print_tree(node.left, depth + 1, column_names)
            print(f"{indent}else:")
            self.print_tree(node.right, depth + 1, column_names)

    def plot_tree(self, figsize=None, column_names=None) -> None:
        """
        Renders the decision tree.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        if self.root is None:
            raise Exception("Tree not trained. Call fit() first.")

        positions: dict = {}
        counter = [0]

        def _assign_positions(node: Optional[Node], depth: int) -> None:
            if node is None:
                return
            _assign_positions(node.left, depth + 1)
            positions[id(node)] = (counter[0], -depth)
            counter[0] += 1
            _assign_positions(node.right, depth + 1)

        _assign_positions(self.root, 0)

        num_nodes = len(positions)
        if figsize is None:
            figsize = (max(10, num_nodes * 1.4), max(6, (self.max_depth + 2) * 1.6))

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_axis_off()

        def _draw_edges(node: Optional[Node]) -> None:
            if node is None or node.is_leaf():
                return

            px, py = positions[id(node)]

            for child, label in ((node.left, "T"), (node.right, "F")):
                if child is not None:
                    cx, cy = positions[id(child)]
                    ax.plot([px, cx], [py, cy], color="#888888", linewidth=1.2, zorder=1)
                    # Place the edge label at 30 % of the way from parent to child
                    mx, my = px + 0.3 * (cx - px), py + 0.3 * (cy - py)
                    ax.text(
                        mx, my, label,
                        ha="center", va="center",
                        fontsize=7, color="#555555",
                        bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none"),
                        zorder=2,
                    )
                _draw_edges(child)

        _draw_edges(self.root)

        def _draw_nodes(node: Optional[Node]) -> None:
            if node is None:
                return

            x, y = positions[id(node)]

            if node.is_leaf():
                label = f"Predict\n{node.prediction}"
                facecolor = "#b7e4c7"  # soft green for leaves
                edgecolor = "#2d6a4f"
            else:
                label = node.condition.get_label(column_names)
                facecolor = "#aecbfa"  # soft blue for internal nodes
                edgecolor = "#1a56a0"

            ax.text(
                x, y, label,
                ha="center", va="center",
                fontsize=8, fontweight="bold" if node is self.root else "normal",
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    linewidth=1.5,
                ),
                zorder=3,
            )

            _draw_nodes(node.left)
            _draw_nodes(node.right)

        _draw_nodes(self.root)

        xs = [p[0] for p in positions.values()]
        ys = [p[1] for p in positions.values()]
        x_margin = max(1, (max(xs) - min(xs)) * 0.05)
        y_margin = 0.8
        ax.set_xlim(min(xs) - x_margin, max(xs) + x_margin)
        ax.set_ylim(min(ys) - y_margin, max(ys) + y_margin)

        # Legend
        legend_handles = [
            mpatches.Patch(facecolor="#aecbfa", edgecolor="#1a56a0", label="Internal node (split)"),
            mpatches.Patch(facecolor="#b7e4c7", edgecolor="#2d6a4f", label="Leaf node (prediction)"),
        ]
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

        plt.title("Decision Tree", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.show()

    # Tree construction

    def _build_tree(
        self,
        data: pd.DataFrame,
        conditions: List[Condition],
        target_index: int,
        depth: int) -> Node:
        """
        Recursively partitions `data` to construct the tree.

        1. If the data is already pure, return a leaf.
        2. Find the best condition by Information Gain.
        3. If no valid split exists or max depth is reached, return a majority-vote leaf.
        4. Otherwise, split and recurse on each partition.

        Conditions are pruned for child nodes: once a LESS_THAN threshold is chosen
        for a feature, thresholds above it are removed from the left child's candidates
        and thresholds below it are removed from the right child's candidates.
        """
        unique_values = data.iloc[:, target_index].unique()
        if len(unique_values) == 1:
            return Node(prediction=unique_values[0])

        best_split, true_examples, false_examples = self._find_best_split(
            data, conditions, target_index
        )

        # Base case 2: no informative split found, or depth limit reached
        if best_split is None or depth >= self.max_depth:
            return Node(
                prediction=self._majority_vote(data, target_index),
                left=None,
                right=None,
                condition=None,
            )

        # Prune the condition list for each child to avoid redundant future splits.
        if best_split.operator == ComparisonOperators.LESS_THAN:
            left_child_conditions = [
                c for c in conditions
                if c.feature_index != best_split.feature_index or c < best_split
            ]
            right_child_conditions = [
                c for c in conditions
                if c.feature_index != best_split.feature_index or c > best_split
            ]
        else:  # EQUALS
            left_child_conditions = [
                c for c in conditions
                if c.feature_index != best_split.feature_index
            ]
            right_child_conditions = [
                c for c in conditions
                if c is not best_split
            ]

        left_node = self._build_tree(true_examples, left_child_conditions, target_index, depth + 1)
        right_node = self._build_tree(false_examples, right_child_conditions, target_index, depth + 1)

        # if both children are leaves with the same prediction collapse them into a single leaf with that prediction.
        if left_node.is_leaf() and right_node.is_leaf() and left_node.prediction == right_node.prediction:
            return Node(prediction=left_node.prediction)

        return Node(condition=best_split, left=left_node, right=right_node)

    def _find_best_split(
        self,
        data: pd.DataFrame,
        conditions: List[Condition],
        target_index: int) -> Tuple[Optional[Condition], pd.DataFrame, pd.DataFrame]:
        """
        Evaluates every candidate condition and returns the one with the highest
        Information Gain above `gamma`.

        Information Gain is computed as:
            IG = H(parent) - [p_true * H(true_split) + p_false * H(false_split)]

        Returns:
            (best_condition, true_partition, false_partition)
            If no condition beats `gamma`, best_condition is None and partitions are empty DataFrames.
        """
        best_gain = 0
        best_condition: Optional[Condition] = None
        best_true_split: pd.DataFrame = pd.DataFrame()
        best_false_split: pd.DataFrame = pd.DataFrame()

        current_impurity = self._entropy(data, target_index)

        for condition in conditions:
            mask = condition.get_mask(data)
            true_examples = data[mask]
            false_examples = data[~mask]

            # Skip conditions that don't split the data
            if len(true_examples) == 0 or len(false_examples) == 0:
                continue

            true_impurity = self._entropy(true_examples, target_index)
            false_impurity = self._entropy(false_examples, target_index)

            # Weighted average impurity of the two resulting partitions
            p_true = len(true_examples) / len(data)
            p_false = 1 - p_true
            weighted_impurity = p_true * true_impurity + p_false * false_impurity

            gain = current_impurity - weighted_impurity

            if gain > best_gain and gain > self.gamma:
                best_gain = gain
                best_condition = condition
                best_true_split = true_examples
                best_false_split = false_examples

        return best_condition, best_true_split, best_false_split

    # Prediction helpers

    def _predict_one(self, row: pd.Series, node: Node) -> int:
        """
        Iteratively traverses the tree for a single data row and returns the predicted label.
        """
        current = node
        while not current.is_leaf():
            mask = current.condition.get_mask(row.to_frame().T)
            current = current.left if mask.iloc[0] else current.right
        return current.prediction

    # Impurity / voting helpers (private)

    def _entropy(self, data: pd.DataFrame, target_index: int) -> float:
        """
        Computes entropy of the target column in `data`.

            H = - sum(p_i * log2(p_i))

        Returns 0 for a pure node.
        """
        counts = data.iloc[:, target_index].value_counts()
        probabilities = counts / len(data)
        entropy = -sum(probabilities * np.log2(probabilities))
        return entropy

    def _majority_vote(self, data: pd.DataFrame, target_index: int) -> Optional[int]:
        """
        Returns the most frequent class label in the target column.
        Returns None if `data` is empty.
        """
        if len(data) == 0:
            return None
        counts = data.iloc[:, target_index].value_counts()
        return counts.idxmax()