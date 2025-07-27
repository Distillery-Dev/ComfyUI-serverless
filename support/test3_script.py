import asyncio
from custom_nodes.discomfort.discomfort import Discomfort

async def main():
    discomfort = await Discomfort.create()
    image = discomfort.Tools.open_image_as_tensor("test_woman.png")
    prompt = "A beautiful scifi woman with long blonde hair and blue eyes, masterpiece"
    model_name = "mohawk.safetensors"
    lora_name = "scifixl.safetensors"

    load_model_and_lora_workflow = "custom_nodes/discomfort/support/sdxl_load_model_and_lora.json"
    latent_empty_workflow = "custom_nodes/discomfort/support/latent_1024x1024_empty.json"
    latent_from_image_workflow = "custom_nodes/discomfort/support/latent_from_input_image.json"
    sampler_workflow = "custom_nodes/discomfort/support/sdxl_run_ksampler.json"

    with discomfort.Context() as context:
        # Save the initial parameters needed by the first workflow
        context.save("prompt", prompt)
        context.save("model_name", model_name)
        context.save("lora_name", lora_name)
        
        for i in range(10):
            context.save("lora_strength", i * 0.1)
            # This populates the context with "model", "clip", and "vae" objects.
            await discomfort.run([load_model_and_lora_workflow], context=context)
            if i % 2 == 0: # empty latent
                denoise = 1
                await discomfort.run([latent_empty_workflow], context=context)
            else: # img2img latent                
                denoise = 0.5
                context.save("input_image", image)
                await discomfort.run([latent_from_image_workflow], inputs={"input_image": image}, context=context)
            await discomfort.run([sampler_workflow], inputs={"denoise": denoise}, context=context) # Run the KSampler
            discomfort.Tools.save_comfy_image_to_disk(context.load("output_image"), f"img_{i}.png") # Save the output image

    await discomfort.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
