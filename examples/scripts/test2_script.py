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

