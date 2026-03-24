from itertools import product

from dataset import NumericDataset
from decision_tree import DecisionTreeClassifier
from evaluator import BinaryClassificationEvaluator

GAMMA = 10e-5

depths          = [2, 3, 4, 5, 6]
max_cuts_list   = [3, 4, 8, 16, 24, 32]
categorical_cols_list = [
    [2, 3],
    [3],
    [2],
    [2, 3, 5, 6, 7, 8, 9, 10],
    [3, 5, 6, 7, 8, 9, 10],
    [2, 5, 6, 7, 8, 9, 10],
]

dataset = NumericDataset("./data.csv", target=-1)

# results[(depth, max_cuts, categorical_cols)] = (accuracy, precision, recall, fpr)
results = {}

total = len(depths) * len(max_cuts_list) * len(categorical_cols_list)
done = 0

for depth, max_cuts, categorical_cols in product(depths, max_cuts_list, categorical_cols_list):
    done += 1
    print(f"[{done}/{total}] depth={depth}, max_cuts={max_cuts}, cat={categorical_cols}")

    clf = DecisionTreeClassifier(max_depth=depth, gamma=GAMMA, max_cuts=max_cuts, categorical_cols=categorical_cols)
    clf.fit(dataset)

    preds_test = clf.predict(dataset)
    Y_test = dataset.test.iloc[:, dataset.target]

    evaluator = BinaryClassificationEvaluator(Y_test, preds_test)

    key = (depth, max_cuts, tuple(categorical_cols))
    results[key] = (evaluator.accuracy(), evaluator.precision(), evaluator.recall(), evaluator.false_positive_rate())

# Find best classifier: lowest FPR among models within 1% of the best recall
RECALL_TOLERANCE = 0.01
best_recall = max(v[2] for v in results.values())
candidates = {k: v for k, v in results.items() if v[2] >= best_recall - RECALL_TOLERANCE}
best_key = min(candidates, key=lambda k: results[k][3])
best = results[best_key]

# Print results table
col_widths = [7, 9, 30, 10, 10, 10, 10]
header = f"{'Depth':>{col_widths[0]}}  {'MaxCuts':>{col_widths[1]}}  {'CatCols':<{col_widths[2]}}  {'Accuracy':>{col_widths[3]}}  {'Precision':>{col_widths[4]}}  {'Recall':>{col_widths[5]}}  {'FPR':>{col_widths[6]}}"
separator = "-" * len(header)

print()
print(separator)
print(header)
print(separator)

for key, (acc, prec, rec, fpr) in sorted(results.items(), key=lambda x: -x[1][0]):
    depth, max_cuts, cat = key
    marker = " *" if key == best_key else ""
    print(
        f"{depth:>{col_widths[0]}}  "
        f"{max_cuts:>{col_widths[1]}}  "
        f"{str(list(cat)):<{col_widths[2]}}  "
        f"{acc:>{col_widths[3]}.4f}  "
        f"{prec:>{col_widths[4]}.4f}  "
        f"{rec:>{col_widths[5]}.4f}  "
        f"{fpr:>{col_widths[6]}.4f}"
        f"{marker}"
    )

print(separator)
print(f"\nBest: depth={best_key[0]}, max_cuts={best_key[1]}, categorical_cols={list(best_key[2])}")
print(f"  Accuracy={best[0]:.4f}  Precision={best[1]:.4f}  Recall={best[2]:.4f}  FPR={best[3]:.4f}")