import asyncio
from custom_nodes.discomfort.discomfort import Discomfort

async def main():
    discomfort = await Discomfort.create()
    image = discomfort.Tools.open_image_as_tensor("test_woman.png")
    prompt = "A beautiful scifi woman with long blonde hair and blue eyes, masterpiece"
    model_name = "mohawk.safetensors"
    lora_name = "scifixl.safetensors"

    load_model_and_lora_workflow = "custom_nodes/discomfort/support/sdxl_load_model_and_lora.json"
    latent_from_image_workflow = "custom_nodes/discomfort/support/latent_from_input_image.json"
    sampler_workflow = "custom_nodes/discomfort/support/sdxl_run_ksampler.json"
    workflows_to_stitch = [load_model_and_lora_workflow, latent_from_image_workflow, sampler_workflow]
    inputs = {
        "prompt": prompt,
        "model_name": model_name,
        "lora_name": lora_name,
        "input_image": image,
        "lora_strength": 0.8,
        "denoise": 0.5
    }
    stitched_workflow = discomfort.Tools.stitch_workflows(workflows_to_stitch)["stitched_workflow"]

    with discomfort.Context() as context:
        await discomfort.run([stitched_workflow], inputs=inputs, context=context) # Run the full workflow
        discomfort.Tools.save_comfy_image_to_disk(context.load("output_image"), f"output_image.png") # Save the output image
        await discomfort.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
