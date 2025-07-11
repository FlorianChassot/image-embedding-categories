import pickle
import time
import json
import random
from collections import defaultdict
from openai import OpenAI

client = OpenAI()

Sample_per_categories = 25

# Load clustered data
with open("output/clustered_data.pkl", "rb") as f:
    clustered_data = pickle.load(f)

# Organize all entries by cluster
cluster_to_entries = defaultdict(list)
for entry in clustered_data:
    cluster_to_entries[entry["cluster"]].append(entry)

# Randomly sample 10 entries per cluster
cluster_descriptions = []

for cluster_id in sorted(cluster_to_entries.keys()):
    all_entries = cluster_to_entries[cluster_id]
    sampled_entries = random.sample(all_entries, min(Sample_per_categories, len(all_entries)))

    results = []
    print(f"Describing {len(sampled_entries)} images for cluster {cluster_id}...")

    for entry in sampled_entries:
        url = entry["url"]
        try:
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "This image is from a Reddit post. Describe what kind of category this image could fit. 10 words maximum."},
                            {"type": "image_url", "image_url": {"url": url}},
                        ],
                    }
                ],
                max_tokens=30,
            )
            description = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error for {url}: {e}")
            description = "Error"

        results.append({
            "url": url,
            "description": description
        })

        time.sleep(1)

    cluster_descriptions.append(results)

# Save descriptions
with open("output/cluster_descriptions.json", "w", encoding="utf-8") as f:
    json.dump(cluster_descriptions, f, ensure_ascii=False, indent=2)

print("Saved to output/cluster_descriptions.json")
