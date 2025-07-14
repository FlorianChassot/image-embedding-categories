import json
import time
from openai import OpenAI

client = OpenAI()

# Load descriptions from previous step
with open("experimentationSource/output/cluster_descriptions.json", "r", encoding="utf-8") as f:
    cluster_descriptions = json.load(f)

cluster_names = []

for cluster_id, entries in enumerate(cluster_descriptions):
    description_list = [entry["description"] for entry in entries if entry["description"] != "Error"]
    description_text = "\n".join(f"- {desc}" for desc in description_list)

    prompt = (
        "Here are 10 short descriptions of images in social media posts "
        "that belong to one cluster. Give me a name of a category most "
        "fitting for the cluster. Only give the name of the category and 1 to 3 words only\n\n" + description_text
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=30,
        )
        category = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error on cluster {cluster_id}: {e}")
        category = "Error"

    cluster_names.append({
        "cluster_id": cluster_id,
        "category": category,
        "descriptions": description_list  # optional, for traceability
    })

    print(f"Cluster {cluster_id}: {category}")
    time.sleep(1.5)

# Save to file
with open("experimentationSource/output/cluster_names.json", "w", encoding="utf-8") as f:
    json.dump(cluster_names, f, ensure_ascii=False, indent=2)

print("Saved category names to experimentationSource/output/cluster_names.json")