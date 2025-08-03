import asyncio
from custom_nodes.discomfort.discomfort import Discomfort

# add this to the ComfyUI folder and run

async def main():

    discomfort = await Discomfort.create()
    prompt = "beautiful scenery nature glass bottle landscape, , purple galaxy bottle,"
    model_name = "mohawk.safetensors"

    with discomfort.Context() as context:
        context.save("prompt", prompt)
        context.save("model_name", model_name)
        await discomfort.run(["test_workflow.json"], context=context)
        output_image = context.load("output_image")
        discomfort.Tools.save_comfy_image_to_disk(output_image, "test.png")
        await discomfort.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
