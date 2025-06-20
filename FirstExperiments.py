from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import torch
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

## image_file = path
## returns the embedding from the model
def extract_embedding(image_file):
    image = Image.open(image_file).convert("RGB")
    inputs = processor(image, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    # pooler output = embedding of the model
    embedding = outputs.pooler_output[0]
    return embedding

folder_path = "images"
embedding_results = []
image_names = []

for filename in os.listdir(folder_path):
    image_names.append(filename)
    fullPath = folder_path+"/"+filename
    embedding = extract_embedding(fullPath)
    embedding_results.append(embedding)

similarity_matrix = cosine_similarity(embedding_results)

## Sometimes similarity matrix will have 1.00001 due to rounding errors
similarity_matrix = np.minimum(similarity_matrix, 1.0)

df = pd.DataFrame(similarity_matrix, index=image_names, columns=image_names)
df.to_csv("output/similarity_matrix_with_labels.csv")

print("Similarity matrix saved with image names.")
print(f"Max value: {np.max(similarity_matrix)}")
print(f"Shape: {similarity_matrix.shape}")

##plot formating
plt.figure(figsize=(8, 8)) 
heatmap = sns.heatmap(df, cmap='coolwarm', annot=False)
plt.title("Similarity Heatmap")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.subplots_adjust(bottom=0.20, left=0.20)
colorbar = heatmap.collections[0].colorbar
colorbar.set_label("Similarity Score")
plt.savefig("output/Similarity_matrix.png")
 
print("Similarity matrix plot saved")