import pickle
from collections import defaultdict, Counter

with open("experimentationSource/output/clip_classification_results.pkl", "rb") as f:
    clip_results = pickle.load(f)

# Build confusion matrix: true_label → predicted_label counts
conf_matrix = defaultdict(Counter)
# Also needed: predicted_label → true_label counts (inverse view)
inverse_conf_matrix = defaultdict(Counter)

for result in clip_results:
    true = result['true_label']
    pred = result['predicted_label']
    conf_matrix[true][pred] += 1
    inverse_conf_matrix[pred][true] += 1

print("=== True Label View (Prediction Accuracy) ===\n")
for true_label in sorted(conf_matrix):
    predictions = conf_matrix[true_label]
    total = sum(predictions.values())
    most_common_pred, correct = predictions.most_common(1)[0]
    incorrect = total - correct
    print(f"Label: {true_label}")
    print(f"  Most predicted as: {most_common_pred}")
    print(f"  Accuracy: {correct}/{total} ({correct/total*100:.2f}%)")
    if incorrect > 0:
        print(f"  Also predicted as:")
        for label, count in predictions.items():
            if label != most_common_pred:
                print(f"    - {label}: {count} ({count/total*100:.2f}%)")
    print()

print("\n=== Predicted Label View (Prediction Purity) ===\n")
for pred_label in sorted(inverse_conf_matrix):
    true_counts = inverse_conf_matrix[pred_label]
    total = sum(true_counts.values())
    dominant_true_label, count = true_counts.most_common(1)[0]
    impurity = total - count
    print(f"Predicted as: {pred_label}")
    print(f"  Most common true label: {dominant_true_label}")
    print(f"  Purity: {count}/{total} ({count/total*100:.2f}%)")
    if impurity > 0:
        print(f"  Also contains:")
        for label, cnt in true_counts.items():
            if label != dominant_true_label:
                print(f"    - {label}: {cnt} ({cnt/total*100:.2f}%)")
    print()