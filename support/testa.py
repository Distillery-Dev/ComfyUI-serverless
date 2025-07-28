import asyncio
from custom_nodes.discomfort.discomfort import Discomfort

async def main():
    discomfort = await Discomfort.create()
    prompt = "A beautiful scifi woman holding a gun and pointing it at the camera, masterpiece"
    model_name = "mohawk.safetensors"

    workflow = "custom_nodes/discomfort/support/testa.json"
    inputs = {
        "prompt": prompt,
        "model_name": model_name,
    }
    
    with discomfort.Context() as context:
        await discomfort.run([workflow], inputs=inputs, context=context) # Run the full workflow
        discomfort.Tools.save_comfy_image_to_disk(context.load("output_image"), f"output_image.png") # Save the output image
        await discomfort.shutdown()

if __name__ == "__main__":
    asyncio.run(main())