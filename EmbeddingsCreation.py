from transformers import ViTImageProcessor, ViTModel
import os
import requests
from PIL import Image
from io import BytesIO
import torch
import pickle


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

folder_path = "data/Reddit_specific_subs_images" 
embedding_data = []

for txt_file in os.listdir(folder_path):
    if txt_file.endswith(".txt"):
        label = os.path.splitext(txt_file)[0]
        file_path = os.path.join(folder_path, txt_file)

        with open(file_path, 'r') as f:
            urls = [line.strip() for line in f.readlines()][:125]

        for url in urls:
            image = download_image(url)
            if image is None:
                continue
            embedding = extract_embedding(image)
            embedding_data.append({
                "label": label,
                "url": url,
                "embedding": embedding
            })

# Save all embeddings and metadata
with open("output/embeddings.pkl", "wb") as f:
    pickle.dump(embedding_data, f)

print("Done. Embeddings saved to embeddings.pkl")
