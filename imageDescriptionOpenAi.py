from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-4.1-mini",
    input=[{
        "role": "user",
        "content": [
            {"type": "input_text", "text": "This image is from a Reddit post. Describe what kind of category this image could fit. 10 words maximum"},
            {
                "type": "input_image",
                "image_url": "https://i.redd.it/3mx55b0aw2ye1.jpeg",
            },
        ],
    }],
)

print(response.output_text)