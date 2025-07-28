import asyncio
from custom_nodes.discomfort.discomfort import Discomfort
import json

async def main():
    discomfort = await Discomfort.create()
    workflow1 = "custom_nodes/discomfort/support/workflow1.json"
    workflow2 = json.load(open("custom_nodes/discomfort/support/workflow2.json"))
    workflow3 = "custom_nodes/discomfort/support/workflow3.json"
    stitched_workflow = discomfort.Tools.stitch_workflows([workflow1, workflow2, workflow3], delete_input_ports=True, delete_output_ports=True)["stitched_workflow"]
    with open("stitched_workflow.json", "w") as f:
        json.dump(stitched_workflow, f, indent=4)
    
if __name__ == "__main__":
    asyncio.run(main())