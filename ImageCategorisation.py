import os
import time
import json
import pickle
import random
import argparse
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from collections import defaultdict
from transformers import ViTImageProcessor, ViTModel
from k_means_constrained import KMeansConstrained
from openai import OpenAI
import torch

# ========================== CONFIGURATION ==========================
INPUT_FILE = "CountOnceADay/data_sample_test"
OUTPUT_FOLDER = "CountOnceADay/output"
SAMPLE_PER_CATEGORY = 25
AMOUNT_OF_CLUSTERS = 8
MIN_CLUSTER_SIZE_RATIO = 0.04

# Load model and processor
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

client = OpenAI()

# ========================== UTILITY FUNCTIONS ==========================

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

# ========================== PIPELINE STEPS ==========================

def create_embeddings():
    print("🚀 Starting embedding...")
    embedding_data = []
    image_counter = 0

    for txt_file in os.listdir(INPUT_FILE):
        if not txt_file.endswith(".txt"):
            continue
        file_path = os.path.join(INPUT_FILE, txt_file)

        with open(file_path, 'r') as f:
            urls = [line.strip() for line in f.readlines()]

        for url in urls:
            image = download_image(url)
            if image is None:
                continue
            embedding = extract_embedding(image)
            embedding_data.append({"url": url, "embedding": embedding})
            image_counter += 1

            if image_counter % 100 == 0:
                print(f"Processed {image_counter} images...")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    with open(os.path.join(OUTPUT_FOLDER, "embeddings.pkl"), "wb") as f:
        pickle.dump(embedding_data, f)

    print(f"✅ Embeddings created for {image_counter} images.")

def create_clusters():
    print("Starting cluster creation")
    with open(os.path.join(OUTPUT_FOLDER, "embeddings.pkl"), "rb") as f:
        data = pickle.load(f)

    embeddings = torch.stack([entry['embedding'] for entry in data]).numpy()
    total_points = len(embeddings)
    min_size = int(MIN_CLUSTER_SIZE_RATIO * total_points)

    clf = KMeansConstrained(
        n_clusters=AMOUNT_OF_CLUSTERS,
        size_min=min_size,
        random_state=42
    )
    clusters = clf.fit_predict(embeddings)

    for i, cluster_id in enumerate(clusters):
        data[i]["cluster"] = int(cluster_id)

    with open(os.path.join(OUTPUT_FOLDER, "clustered_data.pkl"), "wb") as f:
        pickle.dump(data, f)

    print("✅ Clustered data saved.")

def label_entries():
    with open(os.path.join(OUTPUT_FOLDER, "clustered_data.pkl"), "rb") as f:
        clustered_data = pickle.load(f)

    cluster_to_entries = defaultdict(list)
    for entry in clustered_data:
        cluster_to_entries[entry["cluster"]].append(entry)

    cluster_descriptions = []

    for cluster_id in sorted(cluster_to_entries.keys()):
        entries = cluster_to_entries[cluster_id]
        sampled = random.sample(entries, min(SAMPLE_PER_CATEGORY, len(entries)))
        results = []

        print(f"🔍 Describing {len(sampled)} images for cluster {cluster_id}...")

        for entry in sampled:
            url = entry["url"]
            try:
                response = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "This image is from a Reddit post. Describe what kind of category this image could fit. 10 words maximum."},
                            {"type": "image_url", "image_url": {"url": url}},
                        ],
                    }],
                    max_tokens=30,
                )
                description = response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Error for {url}: {e}")
                description = "Error"

            results.append({"url": url, "description": description})
            time.sleep(1)

        cluster_descriptions.append(results)

    with open(os.path.join(OUTPUT_FOLDER, "cluster_descriptions.json"), "w", encoding="utf-8") as f:
        json.dump(cluster_descriptions, f, ensure_ascii=False, indent=2)

    print("✅ Saved image descriptions.")

def label_clusters():
    with open(os.path.join(OUTPUT_FOLDER, "cluster_descriptions.json"), "r", encoding="utf-8") as f:
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
                messages=[{"role": "user", "content": prompt}],
                max_tokens=30,
            )
            category = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error on cluster {cluster_id}: {e}")
            category = "Error"

        cluster_names.append({
            "cluster_id": cluster_id,
            "category": category,
            "descriptions": description_list
        })

        print(f"📦 Cluster {cluster_id}: {category}")
        time.sleep(1.5)

    with open(os.path.join(OUTPUT_FOLDER, "cluster_names.json"), "w", encoding="utf-8") as f:
        json.dump(cluster_names, f, ensure_ascii=False, indent=2)

    print("✅ Saved category names.")

# ========================== MAIN ==========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run image clustering pipeline steps.")
    parser.add_argument("--embed", action="store_true", help="Create image embeddings")
    parser.add_argument("--cluster", action="store_true", help="Create clusters from embeddings")
    parser.add_argument("--entry", action="store_true", help="Label image entries in clusters")
    parser.add_argument("--label", action="store_true", help="Label each cluster with a name")

    parser.add_argument("--input", type=str, default="data", help="Input folder with .txt files")
    parser.add_argument("--output", type=str, default="output", help="Output folder for results")
    parser.add_argument("--sample", type=int, default=25, help="Sample size per cluster for description")
    parser.add_argument("--clusters", type=int, default=8, help="Number of clusters")
    parser.add_argument("--min-cluster-ratio", type=float, default=0.05, help="Minimum ratio for cluster size (0.0 to 1.0)")

    args = parser.parse_args()

    # Assign config from args
    INPUT_FILE = args.input
    OUTPUT_FOLDER = args.output
    SAMPLE_PER_CATEGORY = args.sample
    AMOUNT_OF_CLUSTERS = args.clusters
    MIN_CLUSTER_SIZE_RATIO = args.min_cluster_ratio

    # If no steps selected, run everything
    if not any([args.embed, args.cluster, args.entry, args.label]):
        args.embed = args.cluster = args.entry = args.label = True
        print("🔁 No specific steps provided — running the full pipeline.")

    if(MIN_CLUSTER_SIZE_RATIO*AMOUNT_OF_CLUSTERS>1):
        print("The minimum ratio per cluster * the amount of cluster must be equal or lesser than 1")
        exit(1)

    # Create output folder if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Run selected steps
    if args.embed:
        create_embeddings()
    if args.cluster:
        create_clusters()
    if args.entry:
        label_entries()
    if args.label:
        label_clusters()