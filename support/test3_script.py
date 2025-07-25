import asyncio
from PIL import Image
import torch
import numpy
from custom_nodes.discomfort.discomfort import Discomfort

def get_image_tensor(image_path):
    image = Image.open(image_path)
    image = image.convert("RGB")
    image = numpy.array(image).astype(numpy.float32) / 255.0
    image = torch.from_numpy(image).unsqueeze(0)
    return image

def save_comfy_image(tensor, output_path):
    image = tensor[0]
    image = image * 255
    image = image.clamp(0, 255).to(torch.uint8)
    image_np = image.cpu().numpy()
    pil_image = Image.fromarray(image_np, 'RGB')
    pil_image.save(output_path)

async def main():
    discomfort = await Discomfort.create()
    image = get_image_tensor("custom_nodes/discomfort/support/test_woman.png")
    prompt = "A beautiful scifi woman with long blonde hair and blue eyes, masterpiece"
    model_name = "mohawk.safetensors"
    lora_name = "scifixl.safetensors"

    prepare_workflow = "custom_nodes/discomfort/support/sdxl_load_model_and_clip.json"
    latent_empty_workflow = "custom_nodes/discomfort/support/latent_1024x1024_empty.json"
    latent_from_image_workflow = "custom_nodes/discomfort/support/latent_from_input_image.json"
    sampler_workflow = "custom_nodes/discomfort/support/sdxl_run_ksampler.json"

    with discomfort.Context() as context:
        # Save the initial parameters needed by the first workflow
        context.save("prompt", prompt)
        context.save("model_name", model_name)
        context.save("lora_name", lora_name)
        
        for i in range(10):
            print(f"--- Starting iteration {i} ---")
            context.save("lora_strength", i * 0.1)
            print("--- STEP 1: Load models and encode prompts ---")
            # This populates the context with "model", "clip", and "vae" objects.
            await discomfort.run([prepare_workflow], context=context)
            print("--- STEP 2: Create the latent image ---")
            if i % 2 == 0: # empty latent
                await discomfort.run([latent_empty_workflow], context=context)
                image_suffix = "empty"
            else: # img2img latent                
                await discomfort.run([latent_from_image_workflow], inputs={"input_image": image}, context=context) # This workflow needs the VAE from the previous step, which is already in the context
                image_suffix = "input"
            print("--- STEP 3: Run the sampler ---")
            await discomfort.run([sampler_workflow], context=context)
            print("--- STEP 4: Save the output ---")
            output_tensor = context.load("output_image")
            save_comfy_image(output_tensor, f"custom_nodes/discomfort/support/temp/output_image_{i}_{image_suffix}.png")
            print(f"--- Saved output_image_{i}_{image_suffix}.png ---")

    await discomfort.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
