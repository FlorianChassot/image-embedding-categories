import pickle
from collections import defaultdict, Counter

with open("output/clip_classification_results.pkl", "rb") as f:
    clip_results = pickle.load(f)

conf_matrix = defaultdict(Counter)

for result in clip_results:
    true_label = result['true_label']
    pred_label = result['predicted_label']
    conf_matrix[true_label][pred_label] += 1

print("=== CLIP Classification Evaluation ===\n")

all_correct = 0
all_total = 0

for true_label in sorted(conf_matrix):
    predictions = conf_matrix[true_label]
    total = sum(predictions.values())
    best_guess, correct = predictions.most_common(1)[0]

    all_correct += correct
    all_total += total

    print(f"Label: {true_label}")
    print(f"  Most predicted as: {best_guess} ({correct}/{total}, {correct/total*100:.2f}%)")
    print("  Distribution:")
    for pred_label, cnt in predictions.items():
        print(f"    - {pred_label}: {cnt} ({cnt/total*100:.2f}%)")
    print()

# Overall accuracy
print(f"Overall accuracy: {all_correct}/{all_total} = {(all_correct / all_total) * 100:.2f}%")
