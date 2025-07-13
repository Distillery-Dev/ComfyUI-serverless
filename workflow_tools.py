import json
import networkx as nx  # For topological sort; pip install if needed
import shutil  # For temp dir cleanup
import tempfile  # For temporary storage
import os  # For path operations
import time  # For timing
import torch  # For tensor serialization
import cloudpickle
import base64
from io import BytesIO
import sys  # For sizeof checks
import inspect # For dynamic input discovery

from typing import Dict, List, Any, Optional
import argparse

import numpy as np
from PIL import Image  # For image loading/saving
import logging

try:
    import server  # For PromptServer logging
except ModuleNotFoundError:
    server = None  # Fallback for standalone

import requests  # For API calls if standalone
import uuid
import execution  # For validate_prompt
import asyncio
try:
    import nodes
    NODE_CLASS_MAPPINGS = nodes.NODE_CLASS_MAPPINGS
except ImportError:
    NODE_CLASS_MAPPINGS = {}

# Temporary data store for DiscomfortPort execution (using references, not direct storage)
_temp_execution_data = {}

# Note: Removed monkey patch - using simpler approach by searching all execution stores


class DiscomfortWorkflowTools:
    def log_message(self, message: str, level: str = "info"):
        msg = f"[Discomfort] {level.upper()}: {message}"
        levels = {'debug': logging.DEBUG, 'info': logging.INFO, 'warning': logging.WARNING, 'error': logging.ERROR}
        level_val = levels.get(level.lower(), logging.INFO)
        logging.log(level_val, msg)

    def _get_workflow_with_reroutes_removed(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Returns a new workflow with Reroute nodes removed and links rewired, using the existing utility."""
        
        nodes_dict = {node['id']: node for node in workflow['nodes']}
        links_list = workflow['links']

        clean_nodes_dict, clean_links_list, link_id_map = self.remove_reroute_nodes(nodes_dict, links_list)

        clean_workflow = {
            'nodes': list(clean_nodes_dict.values()),
            'links': clean_links_list,
            **{k: v for k, v in workflow.items() if k not in ['nodes', 'links']}
        }
        
        # Update the 'link' property in each node's inputs using the new map
        for node in clean_workflow['nodes']:
            if 'inputs' in node:
                for node_input in node['inputs']:
                    if 'link' in node_input and node_input['link'] is not None:
                        original_link_id = node_input['link']
                        node_input['link'] = link_id_map.get(original_link_id)
        
        return clean_workflow

    def _build_prompt_from_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        prompt = {}
        nodes_by_id = {n['id']: n for n in workflow.get('nodes', [])}
        links_by_id = {l[0]: l for l in workflow.get('links', [])}

        for node_id, node_data in nodes_by_id.items():
            class_type = node_data['type']
            if class_type not in NODE_CLASS_MAPPINGS:
                self.log_message(f"Node type {class_type} not in NODE_CLASS_MAPPINGS, skipping.", "warning")
                continue
            
            inputs = {}
            
            # Process linked inputs
            linked_input_names = set()
            for node_input in node_data.get('inputs', []):
                if 'link' in node_input and node_input['link'] is not None:
                    link_info = links_by_id.get(node_input['link'])
                    if link_info:
                        source_id, source_slot = link_info[1], link_info[2]
                        inputs[node_input['name']] = [source_id, source_slot]
                        linked_input_names.add(node_input['name'])

            # Process widget inputs
            node_class = NODE_CLASS_MAPPINGS[class_type]
            input_types = node_class.INPUT_TYPES()
            
            widget_inputs = []
            for section in ['required', 'optional']:
                for name, config in input_types.get(section, {}).items():
                    # Heuristic: if it's not a common linkable type, it could be a widget
                    # and it's not already linked
                    if name not in linked_input_names:
                        # The order of widgets_values corresponds to the order of inputs that are widgets.
                        # This iteration preserves the definition order.
                        widget_inputs.append(name)
            
            widget_values = node_data.get('widgets_values', [])
            
            # General-purpose solution to map widget values to the correct inputs
            # by inspecting the node's execution function signature.
            try:
                self.log_message(f"Dynamically mapping widgets for {class_type}...", "debug")
                node_class = NODE_CLASS_MAPPINGS[class_type]
                func_name = node_class.FUNCTION
                # The function can be on the class or an instance of it
                func = getattr(node_class, func_name) if hasattr(node_class, func_name) else getattr(node_class(), func_name)
                sig = inspect.signature(func)
                executable_params = set(sig.parameters.keys())
                self.log_message(f"Found executable params for {class_type}: {executable_params}", "debug")

                # Get all potential widget inputs in their original, stable order, including UI-only extras
                potential_widget_inputs = []
                input_types = node_class.INPUT_TYPES()
                for section in ['required', 'optional']:
                    for name, config in input_types.get(section, {}).items():
                        if name not in linked_input_names:
                            potential_widget_inputs.append(name)
                            # ComfyUI adds a UI-only control widget after seed/noise_seed inputs
                            if name.endswith('seed') or name.endswith('noise_seed'):
                                potential_widget_inputs.append(name + '_control')

                self.log_message(f"Potential widget inputs for {class_type} (in order, with extras): {potential_widget_inputs}", "debug")

                # Consume widget_values and assign them to the widget names
                values = {}
                widget_value_idx = 0
                for input_name in potential_widget_inputs:
                    if widget_value_idx >= len(widget_values):
                        break
                    
                    value = widget_values[widget_value_idx]
                    values[input_name] = value
                    
                    self.log_message(f"Mapped value {value} to widget name {input_name}", "debug")
                    widget_value_idx += 1

                # Now build prompt inputs only for executable params, using the named values
                for param in executable_params:
                    if param in values:
                        inputs[param] = values[param]
                        self.log_message(f"Assigned to prompt inputs: {param} = {values[param]}", "debug")

            except Exception as e:
                self.log_message(f"CRITICAL: Could not dynamically map widgets for {class_type}: {e}. Falling back to simple mapping.", "error")
                # Fallback to the old (potentially buggy) logic if inspect fails
                for i, name in enumerate(widget_inputs):
                    if i < len(widget_values):
                        inputs[name] = widget_values[i]
            
            prompt[node_id] = {'class_type': class_type, 'inputs': inputs}
        
        return prompt

    def validate_workflow(self, workflow: Dict[str, Any]) -> bool:
        required_keys = ['nodes', 'links']
        for key in required_keys:
            if key not in workflow:
                self.log_message(f"Missing required key '{key}' in workflow", "error")
                return False
        # Check link integrity
        node_ids = {n['id'] for n in workflow['nodes']}
        for link in workflow['links']:
            if len(link) != 6:
                self.log_message("Invalid link format", "error")
                return False
            _, src, _, tgt, _, _ = link
            if src not in node_ids or tgt not in node_ids:
                self.log_message(f"Link references invalid node: {src} -> {tgt}", "error")
                return False
        return True

    def discover_ports(self, workflow_path: str) -> Dict[str, Any]:
        try:
            with open(workflow_path, 'r') as f:
                workflow = json.load(f)
            if not self.validate_workflow(workflow):
                raise ValueError("Invalid workflow structure")
            
            nodes = {n['id']: n for n in workflow.get('nodes', [])}
            links = workflow.get('links', [])
            
            # Classify DiscomfortPorts
            inputs = {}
            outputs = {}
            passthrus = {}
            unique_id_to_node = {}
            for node_id, node in nodes.items():
                if node.get('type') == 'DiscomfortPort':
                    wv = node.get('widgets_values', [])
                    unique_id = wv[0] if wv else ''
                    if not unique_id:
                        continue
                    tags = wv[1].split(',') if len(wv) > 1 else []
                    
                    # Incoming links to 'input_data' (assume input slot 0 is 'input_data')
                    incoming = any(link[3] == node_id and link[4] == 0 for link in links)
                    # Outgoing from output slot 0
                    outgoing = any(link[1] == node_id and link[2] == 0 for link in links)
                    
                    # Infer types
                    input_type = 'ANY'
                    output_type = 'ANY'
                    # Incoming (for input_type/upstream)
                    incoming_link = next((l for l in links if l[3] == node_id and l[4] == 0), None)
                    if incoming_link:
                        source_id = incoming_link[1]
                        source_slot = incoming_link[2]
                        source_node = nodes.get(source_id, {})
                        output_types = source_node.get('outputs', [])
                        if source_slot < len(output_types):
                            input_type = output_types[source_slot].get('type', 'ANY') or 'ANY'
                    # Outgoing (for output_type/downstream)
                    outgoing_link = next((l for l in links if l[1] == node_id and l[2] == 0), None)
                    if outgoing_link:
                        target_id = outgoing_link[3]
                        target_slot = outgoing_link[4]
                        target_node = nodes.get(target_id, {})
                        input_types = target_node.get('inputs', [])
                        if target_slot < len(input_types):
                            output_type = input_types[target_slot].get('type', 'ANY') or 'ANY'
                    # Set type: downstream for inputs/passthru, upstream for outputs
                    if not incoming and outgoing:  # Input
                        inferred_type = output_type
                    elif incoming and not outgoing:  # Output
                        inferred_type = input_type
                    elif incoming and outgoing:  # Passthru
                        inferred_type = output_type  # Prioritize downstream
                    else:
                        inferred_type = 'ANY'
                    port_info = {'node_id': node_id, 'tags': tags, 'type': inferred_type, 'input_type': input_type, 'output_type': output_type}
                    if not incoming:
                        inputs[unique_id] = port_info
                    if not outgoing:
                        outputs[unique_id] = port_info
                    if incoming and outgoing:
                        passthrus[unique_id] = port_info
                
                    unique_id_to_node[unique_id] = node_id
            
            # Build graph for topo sort
            graph = nx.DiGraph()
            for node_id in nodes:
                graph.add_node(node_id)
            for link in links:
                if len(link) < 5:
                    continue
                _, source_id, source_slot, target_id, target_slot, _ = link
                graph.add_edge(source_id, target_id)
            
            try:
                execution_order = list(nx.topological_sort(graph))
            except nx.NetworkXUnfeasible:
                raise ValueError('Workflow contains cycles; cannot perform topological sort.')
            except nx.NetworkXError as e:
                raise ValueError(f'Graph error: {e}')
            
            # Infer types
            for unique_id, info in {**inputs, **outputs, **passthrus}.items():
                node_id = info['node_id']
                node = nodes[node_id]
                # For inputs: Type from outgoing target
                for link in links:
                    if link[1] == node_id and link[2] == 0:
                        target_id = link[3]
                        target_slot = link[4]
                        target_node = nodes.get(target_id, {})
                        input_list = target_node.get('inputs', [])
                        if target_slot < len(input_list):
                            info['type'] = input_list[target_slot].get('type', 'ANY')
                        break
                # For outputs: Type from incoming source
                for link in links:
                    if link[3] == node_id and link[4] == 0:
                        source_id = link[1]
                        source_slot = link[2]
                        source_node = nodes.get(source_id, {})
                        output_list = source_node.get('outputs', [])
                        if source_slot < len(output_list):
                            info['type'] = output_list[source_slot].get('type', 'ANY')
                        break
            
            # Add qualifying PASSTHRU to outputs
            for unique_id, info in passthrus.items():
                if unique_id not in outputs:
                    outputs[unique_id] = info
            
            # Enhanced type mismatch check for passthru
            for unique_id, info in passthrus.items():
                if info['input_type'] != info['output_type'] and info['input_type'] != 'ANY' and info['output_type'] != 'ANY':
                    self.log_message(f"Type mismatch in passthru port {unique_id}: {info['input_type']} vs {info['output_type']}", "warning")

            return {
                'inputs': inputs,
                'outputs': outputs,
                'execution_order': execution_order,
                'nodes': nodes,  # Return for merging
                'links': links,
                'unique_id_to_node': unique_id_to_node
            }
        except Exception as e:
            self.log_message(f"Error in discover_ports: {str(e)}", "error")
            raise

    def remove_reroute_nodes(self, nodes, links):
        new_nodes = {nid: n for nid, n in nodes.items() if n['type'] != 'Reroute'}
        new_links = []
        link_id_map = {} # old_id -> new_id
        
        reroute_source_map = {} # reroute_node_id -> (source_node_id, source_slot)
        for node_id, node in nodes.items():
            if node.get('type') == 'Reroute':
                for link in links:
                    if link[3] == node_id:
                        source_id, source_slot = link[1], link[2]
                        # Trace back through multiple reroutes
                        while nodes.get(source_id, {}).get('type') == 'Reroute':
                            found_parent = False
                            for parent_link in links:
                                if parent_link[3] == source_id:
                                    source_id, source_slot = parent_link[1], parent_link[2]
                                    found_parent = True
                                    break
                            if not found_parent:
                                break 
                        reroute_source_map[node_id] = (source_id, source_slot)
                        break
        
        # Keep track of which links have been remapped to avoid duplicates
        processed_links = set()

        for link in links:
            original_link_id = link[0]
            source_id, target_id = link[1], link[3]
            
            if original_link_id in processed_links:
                continue

            # If source is a reroute, replace it with its ultimate source
            if source_id in reroute_source_map:
                final_source_id, final_source_slot = reroute_source_map[source_id]
                new_link = link.copy()
                new_link[1], new_link[2] = final_source_id, final_source_slot
                new_links.append(new_link)
                link_id_map[original_link_id] = new_link[0]
                processed_links.add(original_link_id)

            # If target is a reroute, do nothing, the link going OUT of the target reroute will handle it
            elif target_id in reroute_source_map:
                continue

            # Neither source nor target is a reroute, so keep the link as is
            else:
                new_links.append(link)
                link_id_map[original_link_id] = original_link_id
                processed_links.add(original_link_id)

        return new_nodes, new_links, link_id_map


    def stitch_workflows(self, workflow_paths: List[str]) -> Dict[str, Any]:
        import uuid
        retries = 3
        for attempt in range(retries):
            try:
                if not workflow_paths:
                    raise ValueError('No workflows provided')
                # Load first for base structure
                with open(workflow_paths[0], 'r') as f:
                    base = json.load(f)
                stitched = {
                    'id': str(uuid.uuid4()),
                    'revision': 0,
                    'version': 0.4,
                    'groups': [],
                    'config': {},
                    'extra': base.get('extra', {}),
                    'nodes': [],
                    'links': []
                }
                merged_nodes = stitched['nodes']
                merged_links = stitched['links']
                combined_inputs = {}
                combined_outputs = {}
                node_id_offset = 0
                link_id = max([l[0] for l in base.get('links', [])] or [0]) + 1  # Start after base max
                prev_outputs = {}  # uid: {'node_id': id, 'type': typ}
                for path in workflow_paths:
                    info = self.discover_ports(path)
                    execution_order = info['execution_order']
                    old_to_new_id = {}
                    ordered_nodes = [info['nodes'][n_id] for n_id in execution_order]
                    for idx, node in enumerate(ordered_nodes, start=1):
                        old_id = node['id']
                        new_id = node_id_offset + idx
                        old_to_new_id[old_id] = new_id
                        node['id'] = new_id
                        merged_nodes.append(node)
                    node_id_offset += len(ordered_nodes)
                    # Renumber internal links
                    old_to_new_link = {}
                    for link in info['links']:
                        old_id = link[0]
                        new_link = link.copy()
                        new_link[0] = link_id
                        new_link[1] = old_to_new_id.get(link[1], link[1])
                        new_link[3] = old_to_new_id.get(link[3], link[3])
                        merged_links.append(new_link)
                        old_to_new_link[old_id] = link_id
                        link_id += 1
                    # Update per-node link refs
                    for node in ordered_nodes:
                        for inp in node.get('inputs', []):
                            if 'link' in inp:
                                inp['link'] = old_to_new_link.get(inp['link'], inp['link'])
                        for out in node.get('outputs', []):
                            if 'links' in out:
                                if out['links'] is None:
                                    out['links'] = []
                                out['links'] = [old_to_new_link.get(l, l) for l in out['links']]
                    # Add cross-links from prev outputs to current inputs
                    for uid, in_info in info['inputs'].items():
                        if uid in prev_outputs:
                            source_id = prev_outputs[uid]['node_id']
                            target_id = old_to_new_id[in_info['node_id']]
                            new_link = [link_id, source_id, 0, target_id, 0, prev_outputs[uid]['type'] or 'ANY']
                            merged_links.append(new_link)
                            # Update source node outputs[0]['links']
                            source_node = next((n for n in merged_nodes if n['id'] == source_id), None)
                            if source_node and source_node.get('outputs'):
                                source_node['outputs'][0].setdefault('links', []).append(link_id)
                            # Update target node inputs[0]['link']
                            target_node = next((n for n in merged_nodes if n['id'] == target_id), None)
                            if target_node and target_node.get('inputs'):
                                target_node['inputs'][0]['link'] = link_id
                            link_id += 1
                            # Update to PASSTHRU (but since classifying post-merge, topo will handle)
                    # Update combined I/O (only unconnected)
                    for uid, in_info in info['inputs'].items():
                        if uid not in prev_outputs:  # Only if not connected from prev
                            combined_inputs[uid] = {'node_id': old_to_new_id[in_info['node_id']], 'tags': in_info['tags'], 'type': in_info['type']}
                    for uid, out_info in info['outputs'].items():
                        combined_outputs[uid] = {'node_id': old_to_new_id[out_info['node_id']], 'tags': out_info['tags'], 'type': out_info['type']}
                        prev_outputs[uid] = {'node_id': old_to_new_id[out_info['node_id']], 'type': out_info['type']}
                # Build final graph for topo sort
                graph = nx.DiGraph()
                for node in merged_nodes:
                    graph.add_node(node['id'])
                for link in merged_links:
                    if len(link) != 6:
                        continue
                    link_id, source_id, source_slot, target_id, target_slot, typ = link
                    graph.add_edge(source_id, target_id)
                if not nx.is_directed_acyclic_graph(graph):
                    raise ValueError('Cycle detected in stitched graph')
                global_order = list(nx.topological_sort(graph))
                # Finalize metadata
                stitched['last_node_id'] = max(n['id'] for n in merged_nodes) if merged_nodes else 0
                stitched['last_link_id'] = max(l[0] for l in merged_links) if merged_links else 0

                # After merging, validate
                if not self.validate_workflow(stitched):
                    raise ValueError("Stitched workflow validation failed")

                # Type match check on cross-links
                for uid in combined_inputs:
                    if uid in prev_outputs and prev_outputs[uid]['type'] != combined_inputs[uid]['type'] and combined_inputs[uid]['type'] != 'ANY':
                        self.log_message(f"Type mismatch on cross-link for {uid}: {prev_outputs[uid]['type']} vs {combined_inputs[uid]['type']}", "warning")

                return {'stitched_workflow': stitched, 'inputs': combined_inputs, 'outputs': combined_outputs, 'execution_order': global_order}
            except nx.NetworkXError as e:
                self.log_message(f"Graph error on attempt {attempt+1}: {str(e)}", "error")
                if attempt == retries - 1:
                    raise
            except Exception as e:
                self.log_message(f"Error in stitch_workflows: {str(e)}", "error")
                raise
        raise ValueError("Max retries exceeded for stitching")

    def serialize(self, data: Any, use_ram: bool = True) -> Dict[str, Any]:
        # Memory safeguard
        size = sys.getsizeof(data)
        if size > 1024 * 1024 * 1024:  # 1GB threshold
            self.log_message(f"Large data detected ({size / (1024**3):.2f} GB) - consider disk storage", "warning")
        serialized = {'type': type(data).__name__, 'content': None}
        if isinstance(data, torch.Tensor):
            if not use_ram:  # Offload to CPU if using disk
                data = data.cpu()
            buffer = BytesIO()
            torch.save(data, buffer)
            serialized['content'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
            serialized['type'] = 'TORCH_TENSOR'  # Specific for tensors
        elif isinstance(data, str):
            serialized['content'] = data
            serialized['type'] = 'STRING'
        elif isinstance(data, (int, float, bool)):
            serialized['content'] = data
        elif isinstance(data, (list, dict)):
            try:
                serialized['content'] = json.dumps(data)
                serialized['type'] = 'JSON'
            except:
                pass
        if serialized['content'] is None:  # Fallback to cloudpickle
            serialized['content'] = base64.b64encode(cloudpickle.dumps(data)).decode('utf-8')
            serialized['type'] = 'CUSTOM'
        return serialized

    def deserialize(self, serialized: Dict[str, Any], expected_type: str = 'ANY') -> Any:
        typ = serialized.get('type')
        content = serialized.get('content')
        if typ == 'TORCH_TENSOR':
            buffer = BytesIO(base64.b64decode(content))
            return torch.load(buffer)
        elif typ == 'STRING':
            return content
        elif typ in ('int', 'float', 'bool'):
            return eval(typ)(content)
        elif typ == 'JSON':
            return json.loads(content)
        elif typ == 'CUSTOM':
            return cloudpickle.loads(base64.b64decode(content))
        else:
            raise ValueError(f"Unknown type {typ} for deserialization")
        # Type check/coercion
        if expected_type != 'ANY' and not isinstance(result, eval(expected_type)):
            self.log_message(f"Deserialized type mismatch: expected {expected_type}, got {type(result)}", "warning")
        return result

    def create_validation_placeholder(self, expected_type: str, execution_id: str, uid: str) -> Any:
        """Create JSON-serializable placeholder values for validation that indicate the correct type."""
        # Create a special marker that the DiscomfortPort node will recognize as a placeholder
        placeholder = {
            "__discomfort_placeholder__": True,
            "execution_id": execution_id,
            "uid": uid,
            "expected_type": expected_type
        }
        
        # For specific types, add type hints that ComfyUI validation might use
        if expected_type == 'IMAGE':
            placeholder["__tensor_info__"] = {"shape": [1, 512, 512, 3], "dtype": "float32"}
        elif expected_type == 'LATENT':
            placeholder["__latent_info__"] = {"samples": {"shape": [1, 4, 64, 64]}}
        elif expected_type == 'MODEL':
            placeholder["__model_info__"] = {"model_type": "diffusion"}
        elif expected_type == 'INT':
            placeholder["__value__"] = 0
        elif expected_type == 'FLOAT':
            placeholder["__value__"] = 0.0
        elif expected_type == 'STRING':
            placeholder["__value__"] = ""
        elif expected_type == 'BOOLEAN':
            placeholder["__value__"] = False
        
        return placeholder

    def create_valid_data_for_type(self, expected_type: str, actual_data: Any = None) -> Any:
        """Create valid data objects that will pass ComfyUI validation.
        If actual_data is provided, wrap it appropriately. Otherwise create minimal valid defaults."""
        
        if actual_data is not None:
            # If we have actual data, ensure it's in the right format
            if expected_type == 'IMAGE' and isinstance(actual_data, torch.Tensor):
                # Ensure it's in ComfyUI's expected format: [batch, height, width, channels]
                if actual_data.dim() == 3:  # [H, W, C]
                    actual_data = actual_data.unsqueeze(0)
                elif actual_data.dim() == 2:  # [H, W]
                    actual_data = actual_data.unsqueeze(0).unsqueeze(-1)
                return actual_data
            elif expected_type == 'LATENT' and isinstance(actual_data, dict):
                # Ensure it has the 'samples' key
                if 'samples' not in actual_data:
                    return {"samples": actual_data}
                return actual_data
            else:
                return actual_data
        
        # Create minimal valid defaults for validation
        if expected_type == 'IMAGE':
            # Create a tiny valid image tensor [1, 64, 64, 3]
            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        elif expected_type == 'LATENT':
            # Create a valid latent dict with samples
            return {"samples": torch.zeros((1, 4, 8, 8), dtype=torch.float32)}
        elif expected_type == 'MODEL':
            # Create a minimal object that might pass validation
            class DummyModel:
                def __init__(self):
                    self.model_sampling = None
            return DummyModel()
        elif expected_type == 'CLIP':
            # Create a minimal CLIP object
            class DummyCLIP:
                def __init__(self):
                    self.patcher = None
                    self.cond_stage_model = None
            return DummyCLIP()
        elif expected_type == 'VAE':
            # Create a minimal VAE object
            class DummyVAE:
                def __init__(self):
                    self.first_stage_model = None
            return DummyVAE()
        elif expected_type == 'INT':
            return 0
        elif expected_type == 'FLOAT':
            return 0.0
        elif expected_type == 'STRING':
            return ""
        elif expected_type == 'BOOLEAN':
            return False
        elif expected_type == 'CONDITIONING':
            # Create a valid conditioning tuple
            return ([[torch.zeros((1, 77, 768), dtype=torch.float32), {}]], )
        else:
            # For unknown types, return an empty dict as a safe fallback
            return {}

    async def run_sequential(self, workflow_paths: List[str], inputs: Dict[str, Any], iterations: int = 1, condition_port: Optional[str] = None, use_ram: bool = True, persist_prefix: Optional[str] = None, temp_dir: str = None, server_url: str = 'http://127.0.0.1:8188') -> Dict[str, Any]:
        if not workflow_paths:
            self.log_message("No workflows provided", "error")
            raise ValueError("No workflows provided")
        self.log_message(f'Loaded {len(NODE_CLASS_MAPPINGS)} node classes', 'info')
        use_temp = temp_dir is None and not persist_prefix
        storage_dir = None
        if not use_ram:
            if persist_prefix:
                storage_dir = persist_prefix
                os.makedirs(storage_dir, exist_ok=True)
                self.log_message(f"Using permanent storage: {storage_dir}", "info")
            else:
                storage_dir = temp_dir or tempfile.mkdtemp(prefix="discomfort_")
                self.log_message(f"Using temporary directory: {storage_dir}", "info")
        
        intermediates = {} if use_ram else {}
        loop_inputs = inputs.copy()
        final_outputs = {}
        try:
            for iter_num in range(iterations):
                self.log_message(f"Starting iteration {iter_num + 1}", "info")
                extracted = {}
                for path in workflow_paths:
                    # Load original workflow to preserve structure
                    with open(path, 'r') as f:
                        original_workflow = json.load(f)
                    
                    info = self.discover_ports(path)
                    
                    # Get a clean workflow with reroutes handled *before* building the prompt
                    clean_workflow = self._get_workflow_with_reroutes_removed(original_workflow)
                    
                    # Prepare injection data using temporary storage BEFORE building prompt
                    execution_id = f"exec_{iter_num}_{time.time()}_{uuid.uuid4().hex[:8]}"
                    injected_data = {}
                    for uid, port_info in info['inputs'].items():
                        if uid in loop_inputs:
                            data = loop_inputs[uid]
                            if not use_ram and isinstance(data, str):  # If file path
                                with open(data, 'r') as f:
                                    serialized = json.load(f)
                                data = self.deserialize(serialized, port_info['type'])
                            elif isinstance(data, dict) and 'type' in data:  # Serialized dict
                                data = self.deserialize(data, port_info['type'])
                            expected_type = port_info['type']
                            if expected_type != 'ANY':
                                type_map = {'INT': int, 'FLOAT': float, 'STRING': str, 'BOOLEAN': bool, 'IMAGE': torch.Tensor, 'LATENT': dict, 'MODEL': object, 'CUSTOM': object}
                                check_type = type_map.get(expected_type, object)
                                if not isinstance(data, check_type):
                                    self.log_message(f"Type mismatch for input {uid}: expected {expected_type}, got {type(data)}", "warning")
                            node_id = port_info['node_id']
                            injected_data[uid] = data
                            self.log_message(f"Prepared injection data for node {node_id} (uid: {uid})", "debug")
                    
                    # Build the prompt using the new, robust method on the CLEAN workflow
                    prompt = self._build_prompt_from_workflow(clean_workflow)
                    self.log_message("Successfully built prompt from workflow.", "debug")

                    # NEW APPROACH: Inject actual valid data that will pass validation
                    # while storing the execution context for DiscomfortPort to use
                    if injected_data:
                        # First, set up the execution context
                        _temp_execution_data[execution_id] = {
                            "injected_data": injected_data,
                            "prompt_id": None,  # Will be set after we get the prompt_id
                            "timestamp": time.time()
                        }
                        
                        # Now inject valid data objects into the prompt
                        for uid, port_info in info['inputs'].items():
                            if uid in loop_inputs:
                                node_id_int = port_info['node_id']
                                if node_id_int in prompt and prompt[node_id_int]['class_type'] == 'DiscomfortPort':
                                    expected_type = port_info['type']
                                    actual_data = injected_data.get(uid)
                                    
                                    # Create valid data that will pass validation
                                    valid_data = self.create_valid_data_for_type(expected_type, actual_data)
                                    
                                    # Inject valid data directly into the prompt for validation
                                    # The DiscomfortPort will check _temp_execution_data to see if it should use real data
                                    prompt[node_id_int]['inputs']['input_data'] = valid_data
                                    
                                    self.log_message(f"Injected valid data for node {node_id_int} (uid: {uid}, type: {expected_type})", "debug")

                    unique_id = f'seq_{iter_num}_{os.path.basename(path)}_{time.time()}'  # Use as client_id
                    prompt_id = str(uuid.uuid4())
                    
                    # Update the prompt_id in execution context if we have one
                    if execution_id in _temp_execution_data:
                        _temp_execution_data[execution_id]["prompt_id"] = prompt_id
                    
                    # Debug: Show the final prompt structure (without sensitive data)
                    prompt_summary = {}
                    for node_id, node_data in prompt.items():
                        inputs_summary = {}
                        for inp_name, inp_value in node_data.get('inputs', {}).items():
                            if hasattr(inp_value, 'shape'):  # Tensor
                                inputs_summary[inp_name] = f"<Tensor shape={inp_value.shape}>"
                            elif isinstance(inp_value, dict) and inp_value.get("__discomfort_wrapped__"):
                                # Handle wrapped data in summary
                                inputs_summary[inp_name] = f"<Wrapped {inp_value.get('uid', 'unknown')} type={port_info.get('type', 'unknown') if 'port_info' in locals() else 'unknown'}>"
                            elif isinstance(inp_value, list) and len(inp_value) == 2 and isinstance(inp_value[0], int):  # Link
                                inputs_summary[inp_name] = f"<Link to [{inp_value[0]}, {inp_value[1]}]>"
                            elif isinstance(inp_value, (str, int, float, bool)):
                                inputs_summary[inp_name] = inp_value
                            else:
                                inputs_summary[inp_name] = f"<{type(inp_value).__name__}>"
                        prompt_summary[str(node_id)] = {
                            'class_type': node_data.get('class_type'),
                            'inputs': inputs_summary
                        }
                    self.log_message(f"Final prompt structure: {json.dumps(prompt_summary, indent=2)}", "debug")
                    
                    # Validate the prompt
                    valid, errors, outputs_to_execute, node_errors = await execution.validate_prompt(prompt_id, prompt)
                    if not valid:
                        self.log_message(f"Validation failed with errors: {errors}", "error")
                        
                        # Log details about failing nodes
                        for node_id, error_details in node_errors.items():
                            node_info = prompt.get(int(node_id), {}) # node_errors uses string keys
                            self.log_message(f"Node {node_id} ({node_info.get('class_type', 'unknown')}): {error_details}", "error")
                        
                        raise ValueError(f"Invalid prompt: {errors}")
                    
                    # Handle any remaining validation issues by removing problematic nodes
                    if node_errors:
                        problem_nodes = set()
                        for err_node in node_errors.keys():
                            # We use the clean_workflow's nodes for this check now
                            clean_nodes_map = {n['id']: n for n in clean_workflow['nodes']}
                            if 'PreviewImage' in clean_nodes_map.get(int(err_node), {}).get('type', ''):
                                problem_nodes.add(int(err_node))
                        
                        for node_id in problem_nodes:
                            if node_id in prompt:
                                del prompt[node_id]
                                self.log_message(f"Removed problematic node {node_id}", "debug")
                        
                        # Re-validate if we removed nodes
                        if problem_nodes:
                            valid, errors, outputs_to_execute, node_errors = await execution.validate_prompt(prompt_id, prompt)
                            if not valid:
                                self.log_message(f"Node errors after cleanup: {node_errors}", "error")
                                self.log_message(f"Validation errors after cleanup: {errors}", "error")
                                raise ValueError(f"Invalid prompt after cleanup: {errors}")
                    
                    extra_data = {'client_id': unique_id}
                    number = iter_num
                    
                    self.log_message(f"Queueing prompt {prompt_id} for execution", "debug")
                    server.PromptServer.instance.prompt_queue.put((number, prompt_id, prompt, extra_data, outputs_to_execute))
                    
                    # Wait for execution to complete
                    start_time = time.time()
                    check_interval = 0.1
                    last_status_check = 0
                    
                    while True:
                        current_time = time.time()
                        elapsed = current_time - start_time
                        
                        if elapsed > 600:  # 10 minute timeout
                            self.log_message(f"Timeout after {elapsed:.1f}s waiting for {path}", "error")
                            # Clean up execution data before raising
                            if execution_id in _temp_execution_data:
                                del _temp_execution_data[execution_id]
                            raise TimeoutError(f"Timeout waiting for {path} in iteration {iter_num + 1}")
                        
                        # Log status periodically
                        if current_time - last_status_check > 5:  # Every 5 seconds
                            self.log_message(f"Still waiting for execution... ({elapsed:.1f}s elapsed)", "debug")
                            last_status_check = current_time
                            
                            # Check if prompt is still in queue
                            queue_info = server.PromptServer.instance.get_queue_info()
                            if 'queue_running' in queue_info:
                                running_prompts = [item[1] for item in queue_info['queue_running']]
                                if prompt_id in running_prompts:
                                    self.log_message(f"Prompt {prompt_id} is currently executing", "debug")
                        
                        await asyncio.sleep(check_interval)
                        
                        # Check for completion
                        history = server.PromptServer.instance.prompt_queue.get_history(prompt_id)
                        if history and prompt_id in history:
                            execution_result = history[prompt_id]
                            if 'outputs' in execution_result:
                                self.log_message(f"Execution completed in {elapsed:.1f}s", "info")
                                break
                            elif 'status' in execution_result and execution_result['status']:
                                status = execution_result['status']
                                if status.get('status_str') == 'error':
                                    error_msg = f"Execution failed: {status.get('messages', ['Unknown error'])}"
                                    self.log_message(error_msg, "error")
                                    # Clean up execution data before raising
                                    if execution_id in _temp_execution_data:
                                        del _temp_execution_data[execution_id]
                                    raise RuntimeError(error_msg)
                    
                    # Extract outputs from the execution result
                    execution_outputs = execution_result.get('outputs', {})
                    
                    # Extract outputs from DiscomfortPort OUTPUT nodes
                    for uid, port_info in info['outputs'].items():
                        node_id_int = port_info['node_id']
                        node_id_str = str(node_id_int)
                        
                        # First check if the DiscomfortPort node stored its collected data
                        found_output = False
                        
                        # Check in the execution outputs
                        if node_id_str in execution_outputs:
                            node_output = execution_outputs[node_id_str]
                            if isinstance(node_output, list) and len(node_output) > 0:
                                # DiscomfortPort returns a tuple with one element
                                output_data = node_output[0]
                                if output_data is not None:
                                    serialized = self.serialize(output_data, use_ram)
                                    if use_ram:
                                        intermediates[uid] = serialized
                                    else:
                                        file_path = os.path.join(storage_dir, f"output_{uid}.json")
                                        with open(file_path, 'w') as f:
                                            json.dump(serialized, f)
                                        intermediates[uid] = file_path
                                    extracted[uid] = intermediates[uid]
                                    found_output = True
                                    self.log_message(f"Extracted output for port {uid} from node {node_id_int}", "debug")
                        
                        if not found_output:
                            # Try to get from the DiscomfortPort's collected data if available
                            # This is a fallback mechanism
                            self.log_message(f"No output found for port {uid} (node {node_id_int}) in execution outputs", "warning")
                    
                    # Clean up temporary execution data
                    if execution_id in _temp_execution_data:
                        del _temp_execution_data[execution_id]
                        self.log_message(f"Cleaned up execution data for: {execution_id}", "debug")
                
                loop_inputs.update(extracted)
                final_outputs.update(extracted)
                
                # Check condition
                if condition_port and condition_port in final_outputs:
                    cond_data = final_outputs[condition_port]
                    if not use_ram and isinstance(cond_data, str):
                        with open(cond_data, 'r') as f:
                            serialized = json.load(f)
                        cond_value = self.deserialize(serialized)
                    else:
                        cond_value = self.deserialize(cond_data) if isinstance(cond_data, dict) else cond_data
                    if not bool(cond_value):  # Simple bool check; enhance if needed
                        self.log_message(f"Condition {condition_port} evaluated False; stopping loop", "info")
                        break
        
        except Exception as e:
            self.log_message(f"Error in run_sequential: {str(e)}", "error")
            raise
        finally:
            if use_temp and storage_dir:
                shutil.rmtree(storage_dir)
                self.log_message(f"Cleaned up temp dir: {storage_dir}", "info")
        
        # Return deserialized finals
        deserialized_outputs = {}
        for uid, data in final_outputs.items():
            if use_ram and isinstance(data, dict) and 'type' in data:
                deserialized_outputs[uid] = self.deserialize(data)
            elif not use_ram and isinstance(data, str):
                with open(data, 'r') as f:
                    serialized = json.load(f)
                deserialized_outputs[uid] = self.deserialize(serialized)
            else:
                deserialized_outputs[uid] = data
        return deserialized_outputs

# Example: tools = DiscomfortWorkflowTools(); result = tools.stitch_workflows(['/path1.json', '/path2.json']) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DiscomfortWorkflowTools CLI: Stitch or Run workflows.')
    parser.add_argument('--mode', choices=['stitch', 'run'], default='stitch', help='Mode: stitch (merge workflows) or run (execute with looping).')
    parser.add_argument('--workflows', nargs='+', required=True, help='List of workflow JSON paths.')
    parser.add_argument('--output', default=None, help='Optional path to save stitched JSON (for stitch mode).')
    parser.add_argument('--inputs', type=str, default='{}', help='JSON string of initial inputs for run mode (e.g., {"port_id": "value"}).')
    parser.add_argument('--iterations', type=int, default=1, help='Number of iterations for run mode.')
    parser.add_argument('--condition_port', type=str, default=None, help='Unique ID of condition port for early break in run mode.')
    parser.add_argument('--use_ram', action='store_true', default=True, help='Use in-RAM storage for run mode.')
    parser.add_argument('--persist_prefix', type=str, default=None, help='Prefix for persistent disk storage in run mode.')
    parser.add_argument('--server_url', type=str, default='http://127.0.0.1:8188', help='ComfyUI server URL for run mode API calls if standalone.')
    args = parser.parse_args()
    
    tools = DiscomfortWorkflowTools()
    
    if args.mode == 'stitch':
        result = tools.stitch_workflows(args.workflows)
        print(f'Stitched Inputs: {result["inputs"]}')
        print(f'Stitched Outputs: {result["outputs"]}')
        print(f'Global Execution Order: {result["execution_order"]}')
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result['stitched_workflow'], f, indent=4)
            print(f'Saved stitched workflow to {args.output}')
    elif args.mode == 'run':
        try:
            inputs_dict = json.loads(args.inputs)
            # Preprocess: Load images if file paths provided (CLI hack for testing)
            for key, value in inputs_dict.items():
                if isinstance(value, str) and value.endswith(('.png', '.jpg', '.jpeg')):
                    img = Image.open(value).convert('RGB')
                    tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)  # [B=1, H, W, C]
                    inputs_dict[key] = tensor
        except json.JSONDecodeError:
            print("Invalid --inputs JSON; using empty dict.")
            inputs_dict = {}
        result = asyncio.run(tools.run_sequential(args.workflows, inputs_dict, args.iterations, args.condition_port, args.use_ram, args.persist_prefix, args.server_url))
        print(f'Final Outputs: {result}')
        # Postprocess: Save tensor outputs as images (CLI hack for testing)
        for uid, data in result.items():
            if isinstance(data, torch.Tensor) and data.dim() == 4:  # Assume [B, H, W, C]
                img_array = (data.squeeze(0).cpu().numpy() * 255).astype(np.uint8)  # [H, W, C]
                img = Image.fromarray(img_array)
                save_path = f'output_{uid}.png'
                img.save(save_path)
                print(f'Saved output image for {uid} to {save_path}') 