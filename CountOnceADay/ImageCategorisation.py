from transformers import ViTImageProcessor, ViTModel
import os
import requests
from PIL import Image
from io import BytesIO
import torch
import pickle
from k_means_constrained import KMeansConstrained
import numpy as np
import time
import json
import random
from collections import defaultdict
from openai import OpenAI

INPUT_FILE = "CountOnceADay/data_sample_test"
OUTPUT_FOLDER = "CountOnceADay/output"
SAMPLE_PER_CATEGORY = 25

LABEL_CLUSTER = True
CREATE_EMBEDDING = False
CREATE_CLUSTER = False
LABEL_ENTRY = False

model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

def extract_embedding(image):
    image = image.convert("RGB")
    inputs = processor(image, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.pooler_output[0]

def download_image(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None

if(CREATE_EMBEDDING):

    folder_path = INPUT_FILE
    embedding_data = []

    for txt_file in os.listdir(folder_path):
        if txt_file.endswith(".txt"):
            label = os.path.splitext(txt_file)[0]
            file_path = os.path.join(folder_path, txt_file)

            with open(file_path, 'r') as f:
                urls = [line.strip() for line in f.readlines()]

            for url in urls:
                image = download_image(url)
                if image is None:
                    continue
                embedding = extract_embedding(image)
                embedding_data.append({
                    "url": url,
                    "embedding": embedding
                })

    # Save all embeddings and metadata
    with open(OUTPUT_FOLDER+"/embeddings.pkl", "wb") as f:
        pickle.dump(embedding_data, f)

    print("Embeddings Created.")

if(CREATE_CLUSTER):

    # Load embeddings
    with open(OUTPUT_FOLDER+"/embeddings.pkl", "rb") as f:
        data = pickle.load(f)

    embeddings = torch.stack([entry['embedding'] for entry in data]).numpy()
    print(f"Embeddings shape: {embeddings.shape}")

    # Parameters
    AMOUNT_OF_CLUSTERS = 8
    MIN_CLUSTER_SIZE_RATIO = 0.04

    # Calculate minimum cluster size
    total_points = len(embeddings)
    min_size = int(MIN_CLUSTER_SIZE_RATIO * total_points)

    # Run constrained KMeans
    clf = KMeansConstrained(
        n_clusters=AMOUNT_OF_CLUSTERS,
        size_min=min_size,
        random_state=42
    )
    clusters = clf.fit_predict(embeddings)

    # Summary of cluster sizes
    print("\n=== Cluster Sizes ===")
    for cluster_id in range(AMOUNT_OF_CLUSTERS):
        count = np.sum(clusters == cluster_id)
        print(f"Cluster {cluster_id}: {count} items")

    # Add cluster assignments to data
    for i, cluster_id in enumerate(clusters):
        data[i]["cluster"] = int(cluster_id)

    # Save to file
    with open(OUTPUT_FOLDER+"/clustered_data.pkl", "wb") as f:
        pickle.dump(data, f)

    print("\nSaved clustered data to "+ OUTPUT_FOLDER+"/clustered_data.pkl")

if(LABEL_ENTRY):
    client = OpenAI()
    # Load clustered data
    with open(OUTPUT_FOLDER+"/clustered_data.pkl", "rb") as f:
        clustered_data = pickle.load(f)

    # Organize all entries by cluster
    cluster_to_entries = defaultdict(list)
    for entry in clustered_data:
        cluster_to_entries[entry["cluster"]].append(entry)

    # Randomly sample 10 entries per cluster
    cluster_descriptions = []

    for cluster_id in sorted(cluster_to_entries.keys()):
        all_entries = cluster_to_entries[cluster_id]
        sampled_entries = random.sample(all_entries, min(SAMPLE_PER_CATEGORY, len(all_entries)))

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
    with open(OUTPUT_FOLDER+"/cluster_descriptions.json", "w", encoding="utf-8") as f:
        json.dump(cluster_descriptions, f, ensure_ascii=False, indent=2)

    print("Saved to " + OUTPUT_FOLDER+ "/cluster_descriptions.json")

if(LABEL_CLUSTER):
    client = OpenAI()

    # Load descriptions from previous step
    with open(OUTPUT_FOLDER+"/cluster_descriptions.json", "r", encoding="utf-8") as f:
        cluster_descriptions = json.load(f)

    cluster_names = []

    for cluster_id, entries in enumerate(cluster_descriptions):
        description_list = [entry["description"] for entry in entries if entry["description"] != "Error"]
        description_text = "\n".join(f"- {desc}" for desc in description_list)

        prompt = (
            "Here are short descriptions of images in social media posts "
            "that belong to one cluster. Give me a name of a category most "
            "fitting for the cluster. Only give the name of the category and 1 to 4 words only\n\n" + description_text
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
    with open(OUTPUT_FOLDER+"/cluster_names.json", "w", encoding="utf-8") as f:
        json.dump(cluster_names, f, ensure_ascii=False, indent=2)

    print("Saved category names to"+OUTPUT_FOLDER+"/cluster_names.json")