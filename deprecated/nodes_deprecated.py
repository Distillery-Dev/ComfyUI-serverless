# nodes_deprecated.py - Deprecated node implementations for reference

# DEPRECATION NOTICE:
# This node (DiscomfortBatchWorkflowExecutor) is deprecated as of the project pivot to loop-enabled workflows.
# Reason: The project now focuses on generalizable looping in ComfyUI, with batch processing as a subset of independent iterations.
# This node was partially built for concurrent batch execution but is untested and does not support dependent loops or stateful chaining.
# Use DiscomfortWorkflowTools' run_sequential (evolving to full loop support) instead. Do not register or use in production.
# For revival, integrate with the new loop paradigm in nodes.py.

import concurrent.futures
import copy
import time
import server

from .nodes import any_typ  # Assuming import from main nodes.py if needed

class DiscomfortBatchWorkflowExecutor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workflow_path": ("STRING", {"default": ""}),
                "images": ("IMAGE",),
                "descriptions": ("STRING",),
                "max_concurrent": ("INT", {"default": 1, "min": 1}),
                "output_tags": ("STRING", {"default": "any", "multiline": True}),
            }
        }

    RETURN_TYPES = (any_typ,)
    RETURN_NAMES = ("results",)
    FUNCTION = "execute_batch"
    CATEGORY = "discomfort/executors"

    def execute_batch(self, workflow_path, images, descriptions, max_concurrent, output_tags):
        with open(workflow_path, 'r') as f:
            original_workflow = json.load(f)

        # Find all Port nodes and classify by mode (based on if 'input_data' will have data)
        input_ports = {}
        output_ports = {}
        passthru_ports = {}
        for node in original_workflow.get('nodes', []):
            if node['type'] == 'DiscomfortPort':
                uid = node.get('widgets_values', {}).get('unique_id', '')
                if uid:
                    has_incoming = any(link[3] == 'input_data' and link[2] == node['id'] for link in original_workflow.get('links', []))
                    has_outgoing = any(link[1] == 0 and link[0] == node['id'] for link in original_workflow.get('links', []))
                    tags = node.get('widgets_values', {}).get('tags', '').split(',')
                    if not has_incoming and has_outgoing:
                        input_ports[uid] = (node['id'], tags)
                    elif has_incoming and not has_outgoing:
                        output_ports[uid] = (node['id'], tags)
                    elif has_incoming and has_outgoing:
                        passthru_ports[uid] = (node['id'], tags)

        if not input_ports or not output_ports:
            raise ValueError("Workflow must include DiscomfortPort nodes in both input and output modes")

        requested_tags = output_tags.split(',')
        image_uid = 'image'
        desc_uid = 'description'
        if image_uid not in input_ports or desc_uid not in input_ports:
            raise ValueError("Missing port unique_ids for image/description")

        results = []
        batch_size = images.shape[0] if hasattr(images, 'shape') else len(images)

        def run_single_workflow(idx):
            retries = 3
            for attempt in range(retries):
                try:
                    workflow = copy.deepcopy(original_workflow)

                    # Type inference: Map port ids to connected types
                    port_types = {}
                    links = workflow.get('links', [])
                    for uid, node_id in input_ports.items():
                        input_type = any_typ
                        output_type = any_typ
                        # Find incoming to input_data (for output mode type)
                        for link in links:
                            if link[2] == node_id and link[3] == 0:  # input_data slot
                                source_id = link[0]
                                source_slot = link[1]
                                source_node = next((n for n in workflow['nodes'] if n['id'] == source_id), None)
                                if source_node:
                                    input_type = source_node.get('output', [any_typ])[source_slot] or any_typ
                        # Find outgoing from output (for input mode type)
                        for link in links:
                            if link[0] == node_id and link[1] == 0:
                                target_id = link[2]
                                target_slot = link[3]
                                target_node = next((n for n in workflow['nodes'] if n['id'] == target_id), None)
                                if target_node:
                                    output_type = target_node.get('input', [any_typ])[target_slot] or any_typ
                        port_types[node_id] = {'input': input_type, 'output': output_type}

                    # Inject and patch types
                    for uid, node_id in input_ports.items():
                        for node in workflow['nodes']:
                            if node['id'] == node_id:
                                node['inputs']['input_data'] = images[idx] if uid == image_uid else descriptions[idx]
                                # Patch type in prompt later
                                break

                    prompt = {str(n['id']): {'class_type': n['type'], 'inputs': n.get('inputs', {})} for n in workflow['nodes']}
                    for p_id, p in prompt.items():
                        if p['class_type'] == 'DiscomfortPort':
                            types = port_types.get(int(p_id), {'input': any_typ, 'output': any_typ})
                            p['return_types'] = [types['output']]  # Patch return to match connected
                            # For input, we can't directly patch but validation will see the source's output

                    # Convert to prompt format (ComfyUI expects dict with node IDs as keys)
                    # Add links/extra as needed

                    for p_id, p in prompt.items():
                        if p['class_type'] == 'DiscomfortPort' and p['inputs'].get('skip_type_check'):
                            # Custom handling to bypass type check (e.g., set type to match expected)
                            p['inputs']['force_type'] = any_typ  # Placeholder; adjust based on connected

                    unique_id = f'batch_{idx}'
                    server.PromptServer.instance.queue_prompt(prompt, unique_id)

                    # Wait for completion with timeout
                    start_time = time.time()
                    while True:
                        if time.time() - start_time > 600:  # 10 min timeout
                            raise TimeoutError(f"Workflow {idx} timed out")
                        time.sleep(0.1)
                        history = server.PromptServer.instance.prompt_queue.get_history(unique_id)
                        if history:
                            break

                    extracted = {}
                    # First collect from output_ports
                    for uid, (node_id, node_tags) in output_ports.items():
                        if set(requested_tags).intersection(node_tags):
                            output_data = history['outputs'].get(str(node_id), {})
                            extracted[uid] = output_data.get('collected')
                    # Then add from passthru if no matching output
                    for uid, (node_id, node_tags) in passthru_ports.items():
                        if uid not in output_ports and set(requested_tags).intersection(node_tags):
                            output_data = history['outputs'].get(str(node_id), {})
                            extracted[uid] = output_data.get('collected')

                    return extracted.get(image_uid)  # Primary result
                except MemoryError as e:
                    server.PromptServer.instance.send_sync("log", {"message": f"[Discomfort] ERROR: OOM in workflow {idx} on attempt {attempt+1}: {str(e)}"})
                    if attempt == retries - 1:
                        return None  # Fallback: skip or return empty
                    time.sleep(1)  # Brief pause before retry
                except Exception as e:
                    server.PromptServer.instance.send_sync("log", {"message": f"[Discomfort] ERROR: Failed workflow {idx} on attempt {attempt+1}: {str(e)}"})
                    if attempt == retries - 1:
                        raise
                    time.sleep(1)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            future_results = [executor.submit(run_single_workflow, i) for i in range(batch_size)]
            for future in concurrent.futures.as_completed(future_results):
                try:
                    results.append(future.result())
                except Exception as e:
                    server.PromptServer.instance.send_sync("log", {"message": f"[Discomfort] ERROR: Batch item failed: {str(e)}"})
                    results.append(None)  # Append None for failed

        return (results,)  # Return list for ANY; adjust as needed 