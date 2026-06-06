from ollama import chat

import sys
from PIL import Image


def resize_for_vlm(input_path: str, output_path: str, max_side: int = 768) -> None:
    image = Image.open(input_path).convert("RGB")
    image.thumbnail((max_side, max_side))
    image.save(output_path, quality=90)

def ask_qwen_image(
    image_path: str,
    prompt: str,
    #model: str = "qwen2.5vl:3b",
    #model: str = "gemma4:e2b"
    model: str = "gemma3:4b"
) -> str:
    response = chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [image_path],
            }
        ],
    )

    return response["message"]["content"]


def describe_shelf(image_path: str) -> str:
    prompt = """
Describe the shelf for a blind user.

Divide the answer into:
- top shelf
- middle shelf
- bottom shelf

Mention only product categories you are reasonably confident about.
Do not guess exact brands unless the text is clearly visible.
Keep the answer short and useful for navigation.
"""

    return ask_qwen_image(image_path, prompt)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ask_qwen.py /path/to/picture.jpg")
        sys.exit(1)

    resize_for_vlm(sys.argv[1], "shelf_small.jpg")

    answer = describe_shelf("shelf_small.jpg")
    print(answer)
