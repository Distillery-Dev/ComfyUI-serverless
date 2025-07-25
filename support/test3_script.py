import asyncio
from PIL import Image
import torch
import numpy
from custom_nodes.discomfort.discomfort import Discomfort

def load_comfy_image(image_path):
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
    image = load_comfy_image("test_woman.png")
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
            save_comfy_image(context.load("output_image"), f"img_{i}.png") # Save the output image

    await discomfort.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
