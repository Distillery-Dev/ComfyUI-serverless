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
            await discomfort.run(["custom_nodes/discomfort/support/discomfort_test1.json"], context=context)
    await discomfort.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
