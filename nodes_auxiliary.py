# nodes_auxiliary.py - Supporting utility nodes for Discomfort (e.g., loaders and describers)

import os
import torch
import requests
from PIL import Image
from openai import OpenAI
import json
import numpy as np
import io
import base64

class DiscomfortFolderImageLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "image_count")
    FUNCTION = "load_images"
    CATEGORY = "discomfort/loaders"

    def load_images(self, folder_path):
        if not os.path.isdir(folder_path):
            raise ValueError(f"Folder not found: {folder_path}")

        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        if not image_files:
            raise ValueError(f"No images found in folder: {folder_path}")

        images = []
        for file in image_files:
            img_path = os.path.join(folder_path, file)
            img = Image.open(img_path).convert("RGB")
            img_tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)  # Normalize to [0,1], add batch dim
            images.append(img_tensor)

        batched_images = torch.cat(images, dim=0) if images else torch.zeros((0,))  # Batch them
        return (batched_images, len(images))

class DiscomfortImageDescriber:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "api_url": ("STRING", {"default": "https://api.openai.com/v1/chat/completions"}),
                "api_key": ("STRING", {"default": "", "password": True}),
                "prompt": ("STRING", {"default": "Describe this image in full detail, including colors, objects, style, and mood.", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("descriptions",)
    FUNCTION = "describe_images"
    CATEGORY = "discomfort/describers"

    def describe_images(self, images, api_url, api_key, prompt):
        if images.shape[0] == 0:
            return ("",)

        client = OpenAI(base_url=api_url, api_key=api_key)
        descriptions = []

        for i in range(images.shape[0]):
            img_tensor = images[i]  # Single image tensor (H, W, C)
            img_np = (img_tensor.numpy() * 255).astype(np.uint8)  # Denormalize to [0,255]
            img = Image.fromarray(img_np)

            # Convert to base64 JPEG
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            try:
                response = client.chat.completions.create(
                    model="gpt-4o",  # Assume vision model; can be made configurable
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                                }
                            ]
                        }
                    ],
                    max_tokens=300  # Limit response length
                )
                desc = response.choices[0].message.content.strip()
                descriptions.append(desc)
            except Exception as e:
                descriptions.append(f"Error describing image {i+1}: {str(e)}")

        return ("\n\n".join(descriptions),) 