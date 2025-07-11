from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from PIL import Image
import torch
import os

# Load instruction-tuned model
processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl")
model.eval()

folder_path = "images"
for file in os.listdir(folder_path):
    path = os.path.join(folder_path, file)
    if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
        continue

    image = Image.open(path).convert("RGB")

    prompt = "Say haaa"
    inputs = processor(images=image, text=prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20)

    label = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"{path}: {label}")