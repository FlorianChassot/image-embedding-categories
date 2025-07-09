import pickle
import torch
from sklearn.cluster import KMeans
from collections import defaultdict, Counter
import numpy as np
from k_means_constrained import KMeansConstrained

with open("output/embeddings.pkl", "rb") as f:
    data = pickle.load(f)
embeddings = torch.stack([entry['embedding'] for entry in data]).numpy()
labels = [entry['label'] for entry in data]
print(embeddings.shape)

##Change this for the amount of categories wanted
AMOUNT_OF_CLUSTERS = 8
MIN_CLUSTER_SIZE_RATIO = 0.04
##kmeans = KMeans(n_clusters=AMOUNT_OF_CLUSTERS, random_state=42)
##clusters = kmeans.fit_predict(embeddings)

total_points = len(embeddings)
min_size = int(MIN_CLUSTER_SIZE_RATIO * total_points)  # at least 10% of total in each cluster

clf = KMeansConstrained(
    n_clusters=AMOUNT_OF_CLUSTERS,
    size_min=min_size,
    random_state=42
)

clusters = clf.fit_predict(embeddings)

cluster_to_labels = defaultdict(list)
label_to_clusters = defaultdict(list)

for idx, cluster_id in enumerate(clusters):
    cluster_to_labels[cluster_id].append(labels[idx])
    label_to_clusters[labels[idx]].append(cluster_id)

# --- Cluster summary ---
print("=== Cluster Summary ===\n")
for cluster_id in range(8):
    cluster_labels = cluster_to_labels[cluster_id]
    label_counts = Counter(cluster_labels)
    most_common_label, count = label_counts.most_common(1)[0]
    percentage = count / len(cluster_labels) * 100

    print(f"Cluster {cluster_id}:")
    print(f"  Most frequent label: {most_common_label}")
    print(f"  Frequency: {count}/{len(cluster_labels)} ({percentage:.2f}%)\n")

# --- Label distribution ---
print("=== Label Distribution Across Clusters ===\n")
label_total_counts = Counter(labels)

for label in sorted(set(labels)):
    cluster_counts = Counter(label_to_clusters[label])
    best_cluster, max_count = cluster_counts.most_common(1)[0]
    total = label_total_counts[label]
    percentage = max_count / total * 100

    print(f"Label: {label}")
    print(f"  Most common in cluster: {best_cluster}")
    print(f"  In cluster: {max_count}/{total} ({percentage:.2f}%)")

    # Misplaced counts
    other_clusters = [(cid, cnt) for cid, cnt in cluster_counts.items() if cid != best_cluster]
    if other_clusters:
        print("  Also found in:")
        for cid, cnt in sorted(other_clusters, key=lambda x: -x[1]):
            print(f"    - Cluster {cid}: {cnt} ({cnt/total*100:.2f}%)")
    print()

    # Add cluster assignments to data
for i, cluster_id in enumerate(clusters):
    data[i]["cluster"] = int(cluster_id)  # Ensure it's JSON serializable if needed

# Save to new file
with open("output/clustered_data.pkl", "wb") as f:
    pickle.dump(data, f)

print("Saved clustered data to output/clustered_data.pkl")