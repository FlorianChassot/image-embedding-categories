from transformers import CLIPProcessor, CLIPModel
import torch
import pickle
from PIL import Image
import requests
from tqdm import tqdm

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Category labels
categories = [
    "Fantasy Dragons", "Humor & Memes", "Digital Collectible Card Games",
    "Motorcycle Lifestyle", "Portrait & Fashion", "Anime and Fantasy Art",
    "Outdoor Adventure & Urban Life", "Urban Cityscapes"
]
text_inputs = clip_processor(text=categories, return_tensors="pt", padding=True)
with torch.no_grad():
    text_features = clip_model.get_text_features(**text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

with open("output/embeddings.pkl", "rb") as f:
    data = pickle.load(f)

clip_results = []

print("Classifying images with CLIP...")

for entry in tqdm(data):
    try:
        image = Image.open(requests.get(entry['url'], stream=True).raw).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt", padding=True)

        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Chose the most similar vector to the image
            similarity = (image_features @ text_features.T)[0]
            best_idx = similarity.argmax().item()
            assigned_label = categories[best_idx]

            clip_results.append({
                "true_label": entry["label"],
                "url": entry["url"],
                "predicted_label": assigned_label
            })
    except Exception as e:
        print(f"Error processing image from {entry['url']}: {e}")

with open("output/clip_classification_results.pkl", "wb") as f:
    pickle.dump(clip_results, f)

print("Done. Saved to output/clip_classification_results.pkl")