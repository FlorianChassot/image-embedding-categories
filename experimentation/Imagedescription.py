from PIL import Image

from transformers import Blip2Processor, Blip2ForConditionalGeneration

import torch

import os 
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

model = Blip2ForConditionalGeneration.from_pretrained(

    "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16

) 
folder_path="images"
for txt_file in os.listdir(folder_path):
    path = folder_path + "/" + txt_file

    image = Image.open(path)

    prompt = "Question: Describe the type of image. Is it a technical document, a photo, a digital art, or something else? Answer:"

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device="cuda", dtype=torch.float16)

    generated_ids = model.generate(**inputs)

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    print(txt_file+ " : " + generated_text)
