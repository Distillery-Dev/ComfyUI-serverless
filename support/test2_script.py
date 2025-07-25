import asyncio
from PIL import Image
import torch
import numpy
from custom_nodes.discomfort.discomfort import Discomfort

# add this to the ComfyUI folder and run

def get_image_tensor(image_path):
    image = Image.open(image_path)
    image = image.convert("RGB")
    image = numpy.array(image).astype(numpy.float32) / 255.0 # Convert to numpy, normalize to 0-1, and set type to float32
    image = torch.from_numpy(image).unsqueeze(0) # Convert to a tensor and add the batch dimension
    return image

async def main():

    discomfort = await Discomfort.create()
    image = get_image_tensor("custom_nodes/discomfort/support/test_woman.png")

    with discomfort.Context() as context:
        seed = 1000
        context.save("input_image", image)
        for i in range(10):
            context.save("seed", seed)
            print(f"--- Iteration {i+1}: Running with SEED = {context.load("seed")} ---")
            seed = seed + i            
            await discomfort.run(["custom_nodes/discomfort/support/discomfort_test2.json"], context=context)
    await discomfort.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
