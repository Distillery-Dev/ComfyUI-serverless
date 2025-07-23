# MAJOR REFACTORING: ADDING PASS-BY-REFERENCE CAPABILITY TO DISCOMFORT 

At this time, Discomfort fails to pass model checkpoints, clip models, etc. via the WorkflowContext object. This behavior is due to our strategy of using cloudpickle for any `unique_id` in WorkflowContext, which ultimately fails to work with complex items like checkpoins/clip models/vae/etc.

Take a look at the workflow discomfort_sdxl_model.json, which I am using in an internal test. It only has 3 unique_id outputs ("model", "clip", "vae"), all of the sort that I'm having trouble with. Indeed when I attempt to run it as a standalone workflow, the logs indicate the workflow fails around the time the element is being saved to context.

The natural way to resolve this is to find a "pass-by-reference" method that could substitute the "pass-by-value" method (ie cloudpickle) we are currently employing, and to design a method to properly choose between one or the other. 

There is an elegant way to fix this issue, one which we already have in our hands: by leveraging on the `stitch_workflows()` method. Since we are building Discomfort on top of ComfyUI, the pass-by-reference method is already in front of us: **the workflow itself that leads to the `unique_id` we intend to save to/load from context**. Instead of passing the `unique_id` data itself, we pass the minimal workflow that leads to it creation, stitching it to whatever subsequent workflow that calls that `unique_id` in an INPUT DiscomfortPort. The `stitch_workflows()` process will transform both OUTPUT and INPUT DiscomfortPorts into PASSTHRU DiscomfortPorts, thus eliminating the data I/O problem.

Thus, for any `unique_id` element that is too complex to handle with cloudpickle, the save() and load() methods must provide, instead of the element itself, the JSON script representative of a workflow JSON, extracted from the original workflow, that should only output that `unique_id` -- that is, one that ONLY contains the nodes that lead up to the `unique_id` element required to be extracted.

There are three challenges to this undertaking, which I'll be addressing below.


## 1. Discovering which `unique_id` requires the `_prune_workflow_to_output()` method

The most general-purpose solution to the issue of discovering which `unique_id` element would require the _prune_workflow_to_output() method would be to capture an error in the cloudpickle attempt, but this will not work due to the points (A) and (B) above. Instead, **we must rely on the `inferred_type` of the `unique_id` as discovered by discover_ports(), and on a predermined set of `inferred_type` that must be passed by reference**.

We should add this set as a new json config file called "pass_by_rules.json". Each key should be a ComfyUI type (ex: "IMAGE", "STRING", "ANY" etc), ideally being an exhaustive list of all ComfyUI types, and each key can be associated with either one of these strings: "val" or "ref". This information then is stored within the `_key_closet` dictionary inside the `WorkflowContext`, under the `pass_by` key for the corresponding `unique_id`. 

For example: if discover_ports() find three OUTPUT DiscomfortPorts -- one being STRING, another being IMAGE, and another being MODEL --, then the WorkflowContext save logic must first check the "pass_by_rules.json" and then discover that STRING and IMAGE must be passed by value (ie both were keys to a "val" string), whereas MODEL is passed by reference (ie MODEL was key to a "ref" string). The information then is (over)written in the `_key_closet` as the `pass_by` key for the corresponding `unique_id`.

Notice that no material changes are needed in WorkflowContext other than the addition of a new key in the `_key_closet` and the logic under save() that consults the `pass_by_rules.json` to figure out whether the `_key_chain` should contain "val" or "ref" for a given `unique_id`. Afterall, whatever is labeled as "ref" will necessarily be a JSON object, therefore compatible with using cloudpickle for storage.  


## 2. Pruning the workflows to their minimal design

We must create a method `_prune_workflow_to_output()`, to be hosted inside WorkflowTools, that receives a workflow and outputs the minimal workflow that lead up to the required `unique_id` OUTPUT DiscomfortPort. We can achieve this by deleting all nodes that are not upstream of the `unique_id` DiscomfortPort node,  through an algorithm that goes through all the nodes of the workflow.json and deletes all unnecessary nodes.


## 3. Properly stitching the workflows together

Once all the of the `unique_id` items whose `pass_by` rule is "ref" were processed by `_prune_workflow_to_output()`, we will end with a set of minimal workflows `ref_workflows`, containing one workflow JSON for each of the `unique_id` items. 

Then, as a step preceding the `_get_prompt_from_workflow()` method, we must stitch all `ref_workflows` to the workflow currently being processed. The stitching process converts the INPUT DiscomfortPorts into PASSTHRU DiscomfortPorts, thus avoiding them being replaced by DiscomfortContextLoaders (which would end up breaking the run).    

(Sidenote: There is a good chance we will end up with several equal, or almost equal, workflows, with several repeated nodes that could eventually be merged together -- for example: LoadCheckpoint nodes offer `model`, `clip` and `vae` outputs, so if each of these were a `unique_id`, then the same LoadCheckpoint node would show up in the final workflow three times. However, we don't need to worry about this issue because ComfyUI's execution is very optimized and handles redundant nodes quite well.)