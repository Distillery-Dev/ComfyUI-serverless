# 🚀 Discomfort: Examples and Usage

## Creating your Discomfort workflows

Discomfort workflows are just regular ComfyUI workflows, but with added DiscomfortPort nodes to signal their inputs and outputs.

To prepare a Discomfort workflow: 
1. Open your workflow
2. Insert a DiscomfortPort node
3. Connect it to whatever you'd like to handle via code
4. Write a **unique_id** inside the DiscomfortPort node
5. Repeat 2-4 until all the nodes you'd like to control are connected to 

**IMPORTANT:**

(a) Make sure connect DiscomfortPorts in the correct way: 
- if you're passing an input via code, then the DiscomfortPort is an INPUT for your workflow and therefore it should START your workflow (connect things on its right side).
- if you're collecting an output to code, then the DiscomfortPort is an OUTPUT for your workflow and therefore it should END your workflow (connect things on its left side).

(b) Discomfort handles partial workflows. This means you can extract a group of nodes from a workflow, add some DiscomfortPorts to it and then use Discomfort to stitch all your workflows together. Stop building spaghetti workflows!

(c) DO NOT connect things to both sides of a DiscomfortPort. If you do that, then it becomes an inactive PASSTHRU node and will NOT interact with Discomfort. This is important for Discomfort's architecture and must be kept this way.

(d) Remember: Discomfort is designed to handle **any type of input/output**. Want to manage a model? The prompt? Save latents? Handle some obscure command in a random node? Go nuts!

## Running your Discomfort script

The `Discomfort` class provides a clean, unified API for all ComfyUI automation. All you need to do is initiate a context, save to/load from it, and then call `Discomfort.run()` with your Discomfort workflows.

See examples below:

```python
# Simple Test Script: add this to the ComfyUI folder and run
# In this simple example, a test workflow iterates over CFG and seed

import asyncio
from custom_nodes.discomfort.discomfort import Discomfort

async def main():

    discomfort = await Discomfort.create()

    with discomfort.Context() as context:
        cfg = 4.0
        seed = 42069
        for i in range(8):
            print(f"--- Iteration {i+1}: Running with CFG = {cfg:.1f} and SEED = {seed} ---")
            cfg = cfg + i*0.2
            seed = seed + i
            context.save("cfg", cfg)
            context.save("seed", seed)
            await discomfort.run(["custom_nodes/discomfort/discomfort_test.json"], context=context)
    await discomfort.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

```python
# Test Script: add this to the ComfyUI folder and run
# In this example, we run different seeds on an img2img workflow

import asyncio
from custom_nodes.discomfort.discomfort import Discomfort

async def main():

    discomfort = await Discomfort.create()
    image = discomfort.Tools.open_image_as_tensor("custom_nodes/discomfort/support/test_woman.png")

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
```

```python
# Test Script: add this to the ComfyUI folder and run
# In this example, we stitch workflows first, run the img2img workflow, then save to disk
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
```

### Advanced Example

```python
# ----------------------------------------------------------------------------------------------------
# Example of code with iterations, conditionals, etc. using 4 different workflows, send to/collect 
# from context models, loras, clip, latent, image and float variables, using partial workflows.
# Fully functional. Run it from the ComfyUI folder.
# ----------------------------------------------------------------------------------------------------
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
```