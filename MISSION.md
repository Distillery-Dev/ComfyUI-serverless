Read all these documents with extreme attention, starting in README.md.

Now we will write together the CURRENT_ISSUES.md. Inside it, we must create all the tasks associated with the demand below.

What I wish to focus on is the following: I want to encapsulate all these methods inside a master class, called Discomfort, in an intuitive and developer-friendly way.

Ultimately my intention is for users to be 100% able to run ComfyUI via Python code. I am with the belief that, with just a little bit of code cleanup, we can make this happen, and making the whole code work under a single Discomfort instance call is the most appropriate way.

Here's what we must do:
- Create another class, called "Discomfort"
- Discomfort must have three subclasses: Tools (renamed version of WorkflowTools), Context (renamed version of WorkflowContext), and Worker (renamed version of ComfyConnector).
- A version of the method run_sequential (basically the code under the `with` statement, before context is created) must exist inside the Discomfort class, as a Discomfort.run() method.

For the avoidance of doubt, I want to be able to run something like the code below:

```python
# ----------------------------------------------------------------------------------------------------
# Example of code that uses some image enhancement workflow, and another that asks a vision LLM to rate the quality of the image
# The idea is that the enhancement would only stop once the vision LLM is satisfied with its quality (>70%)
# ----------------------------------------------------------------------------------------------------
from comfyui-discomfort import Discomfort

discomfort = Discomfort.create()
image = Image.open("initial_image.png")
with discomfort.Context() as context: # create the context of the run
    context.save(image, "input_image") # save the initial_image.png to context as the "input_image" `unique_id`
    quality_index = 0 # initial value so the first iteration always runs
    while context.load("quality_index") < 0.7:
        discomfort.run("workflow_that_improves_images.json", inputs = {"input_image": image}) # Assume this workflow improves an image somehow
        image = context.load("output_image")  # loads the output image after the workflow run
        run("workflow_that_assesses_quality_of_image.json", inputs = {"llm_prompt": "Rate the quality of this image from 0 to 100%.", "image_to_assess": image}) # Assume this workflow is calling a vision LLM and getting it to rate the image from 0-100%
        image = context.load("image_to_assess") # loads the image from the context for returning it OR for the next iteration
        if context.load("quality_index") > 0.7:
            return image
```

Please help me build the task plan for this.