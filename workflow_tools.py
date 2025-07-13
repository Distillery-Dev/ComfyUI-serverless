import json
import networkx as nx
import shutil
import tempfile
import os
import time
import torch
import cloudpickle
import base64
from io import BytesIO
import sys
import inspect
import copy

from typing import Dict, List, Any, Optional
import argparse

import numpy as np
from PIL import Image
import logging

try:
    import server
except ModuleNotFoundError:
    server = None

import requests
import uuid
import execution
import asyncio
try:
    import nodes
    NODE_CLASS_MAPPINGS = nodes.NODE_CLASS_MAPPINGS
except ImportError:
    NODE_CLASS_MAPPINGS = {}

# Import the internal data storage functions
from .nodes_internal import store_data, clear_data, clear_all_memory_data


class DiscomfortWorkflowTools:
    def log_message(self, message: str, level: str = "info"):
        msg = f"[Discomfort] {level.upper()}: {message}"
        levels = {'debug': logging.DEBUG, 'info': logging.INFO, 'warning': logging.WARNING, 'error': logging.ERROR}
        level_val = levels.get(level.lower(), logging.INFO)
        logging.log(level_val, msg)

    def _get_workflow_with_reroutes_removed(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Returns a new workflow with Reroute nodes removed and links rewired."""
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
        """Build a ComfyUI prompt from a workflow."""
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
                    if name not in linked_input_names:
                        widget_inputs.append(name)
            
            widget_values = node_data.get('widgets_values', [])
            
            # Map widget values to inputs
            try:
                self.log_message(f"Dynamically mapping widgets for {class_type}...", "debug")
                func_name = node_class.FUNCTION
                func = getattr(node_class, func_name) if hasattr(node_class, func_name) else getattr(node_class(), func_name)
                sig = inspect.signature(func)
                executable_params = set(sig.parameters.keys())

                # Get all potential widget inputs
                potential_widget_inputs = []
                for section in ['required', 'optional']:
                    for name, config in input_types.get(section, {}).items():
                        if name not in linked_input_names:
                            potential_widget_inputs.append(name)
                            if name.endswith('seed') or name.endswith('noise_seed'):
                                potential_widget_inputs.append(name + '_control')

                # Consume widget_values and assign them
                values = {}
                widget_value_idx = 0
                for input_name in potential_widget_inputs:
                    if widget_value_idx >= len(widget_values):
                        break
                    
                    value = widget_values[widget_value_idx]
                    values[input_name] = value
                    widget_value_idx += 1

                # Build prompt inputs only for executable params
                for param in executable_params:
                    if param in values:
                        inputs[param] = values[param]

            except Exception as e:
                self.log_message(f"Could not dynamically map widgets for {class_type}: {e}. Falling back.", "error")
                for i, name in enumerate(widget_inputs):
                    if i < len(widget_values):
                        inputs[name] = widget_values[i]
            
            prompt[node_id] = {'class_type': class_type, 'inputs': inputs}
        
        return prompt

    def _inject_data_loaders(self, workflow: Dict[str, Any], port_data: Dict[str, Any], 
                           use_ram: bool, storage_dir: Optional[str]) -> Dict[str, Any]:
        """Replace DiscomfortPort INPUT nodes with DiscomfortDataLoader nodes."""
        modified_workflow = copy.deepcopy(workflow)
        nodes_to_remove = []
        nodes_to_add = []
        links_to_update = []
        
        # Find the highest node ID and link ID
        max_node_id = max([n['id'] for n in modified_workflow['nodes']], default=0)
        max_link_id = max([l[0] for l in modified_workflow.get('links', [])], default=0)
        
        # Process each node
        for node_idx, node in enumerate(modified_workflow['nodes']):
            if node.get('type') == 'DiscomfortPort':
                # Get the unique_id from widgets_values
                wv = node.get('widgets_values', [])
                unique_id = wv[0] if wv else ''
                
                # Check if this is an INPUT port (no incoming connections)
                node_id = node['id']
                has_incoming = any(link[3] == node_id and link[4] == 0 for link in modified_workflow.get('links', []))
                
                if not has_incoming and unique_id in port_data:
                    # This is an INPUT port with data to inject
                    self.log_message(f"Replacing DiscomfortPort INPUT '{unique_id}' with DiscomfortDataLoader", "debug")
                    
                    # Store the data
                    data = port_data[unique_id]
                    storage_key = f"{unique_id}_{time.time()}_{uuid.uuid4().hex[:8]}"
                    
                    if use_ram:
                        storage_type = "memory"
                        store_data(storage_key, data, "memory")
                    else:
                        storage_type = "disk"
                        serialized = self.serialize(data, use_ram=False)
                        file_path = os.path.join(storage_dir, f"{storage_key}.json")
                        with open(file_path, 'w') as f:
                            json.dump(serialized, f)
                        storage_key = file_path
                    
                    # Create DiscomfortDataLoader node
                    new_node = {
                        'id': node_id,  # Keep the same ID
                        'type': 'DiscomfortDataLoader',
                        'pos': node.get('pos', [0, 0]),
                        'size': node.get('size', [200, 100]),
                        'flags': node.get('flags', {}),
                        'order': node.get('order', 0),
                        'mode': node.get('mode', 0),
                        'inputs': [],
                        'outputs': node.get('outputs', []),
                        'properties': {},
                        'widgets_values': [storage_key, storage_type, "ANY"]  # storage_key, storage_type, expected_type
                    }
                    
                    # Mark for replacement
                    nodes_to_remove.append(node_idx)
                    nodes_to_add.append((node_idx, new_node))
        
        # Apply modifications
        # Remove nodes in reverse order to maintain indices
        for idx in sorted(nodes_to_remove, reverse=True):
            del modified_workflow['nodes'][idx]
        
        # Add new nodes
        for idx, new_node in sorted(nodes_to_add):
            modified_workflow['nodes'].insert(idx, new_node)
        
        return modified_workflow

    def validate_workflow(self, workflow: Dict[str, Any]) -> bool:
        """Validate workflow structure."""
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
        """Discover DiscomfortPort nodes in a workflow."""
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
                    
                    # Check connections
                    incoming = any(link[3] == node_id and link[4] == 0 for link in links)
                    outgoing = any(link[1] == node_id and link[2] == 0 for link in links)
                    
                    # Infer types
                    input_type = 'ANY'
                    output_type = 'ANY'
                    
                    # Get types from connections
                    incoming_link = next((l for l in links if l[3] == node_id and l[4] == 0), None)
                    if incoming_link:
                        source_id = incoming_link[1]
                        source_slot = incoming_link[2]
                        source_node = nodes.get(source_id, {})
                        output_types = source_node.get('outputs', [])
                        if source_slot < len(output_types):
                            input_type = output_types[source_slot].get('type', 'ANY') or 'ANY'
                    
                    outgoing_link = next((l for l in links if l[1] == node_id and l[2] == 0), None)
                    if outgoing_link:
                        target_id = outgoing_link[3]
                        target_slot = outgoing_link[4]
                        target_node = nodes.get(target_id, {})
                        input_types = target_node.get('inputs', [])
                        if target_slot < len(input_types):
                            output_type = input_types[target_slot].get('type', 'ANY') or 'ANY'
                    
                    # Classify port
                    if not incoming and outgoing:  # Input
                        inferred_type = output_type
                    elif incoming and not outgoing:  # Output
                        inferred_type = input_type
                    elif incoming and outgoing:  # Passthru
                        inferred_type = output_type
                    else:
                        inferred_type = 'ANY'
                    
                    port_info = {
                        'node_id': node_id, 
                        'tags': tags, 
                        'type': inferred_type, 
                        'input_type': input_type, 
                        'output_type': output_type
                    }
                    
                    if not incoming:
                        inputs[unique_id] = port_info
                    if not outgoing:
                        outputs[unique_id] = port_info
                    if incoming and outgoing:
                        passthrus[unique_id] = port_info
                    
                    unique_id_to_node[unique_id] = node_id
            
            # Build graph for topological sort
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
            
            return {
                'inputs': inputs,
                'outputs': outputs,
                'execution_order': execution_order,
                'nodes': nodes,
                'links': links,
                'unique_id_to_node': unique_id_to_node
            }
        except Exception as e:
            self.log_message(f"Error in discover_ports: {str(e)}", "error")
            raise

    def remove_reroute_nodes(self, nodes, links):
        """Remove reroute nodes and rewire connections."""
        new_nodes = {nid: n for nid, n in nodes.items() if n['type'] != 'Reroute'}
        new_links = []
        link_id_map = {}
        
        reroute_source_map = {}
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
        
        # Process links
        processed_links = set()
        for link in links:
            original_link_id = link[0]
            source_id, target_id = link[1], link[3]
            
            if original_link_id in processed_links:
                continue

            if source_id in reroute_source_map:
                final_source_id, final_source_slot = reroute_source_map[source_id]
                new_link = link.copy()
                new_link[1], new_link[2] = final_source_id, final_source_slot
                new_links.append(new_link)
                link_id_map[original_link_id] = new_link[0]
                processed_links.add(original_link_id)
            elif target_id in reroute_source_map:
                continue
            else:
                new_links.append(link)
                link_id_map[original_link_id] = original_link_id
                processed_links.add(original_link_id)

        return new_nodes, new_links, link_id_map

    def serialize(self, data: Any, use_ram: bool = True) -> Dict[str, Any]:
        """Serialize data for storage."""
        size = sys.getsizeof(data)
        if size > 1024 * 1024 * 1024:  # 1GB threshold
            self.log_message(f"Large data detected ({size / (1024**3):.2f} GB) - consider disk storage", "warning")
        
        serialized = {'type': type(data).__name__, 'content': None}
        
        if isinstance(data, torch.Tensor):
            if not use_ram:
                data = data.cpu()
            buffer = BytesIO()
            torch.save(data, buffer)
            serialized['content'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
            serialized['type'] = 'TORCH_TENSOR'
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
        """Deserialize data from storage."""
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

    def stitch_workflows(self, workflow_paths: List[str]) -> Dict[str, Any]:
        """Stitch multiple workflows together."""
        # Implementation remains the same as original
        # ... (keeping the original stitch_workflows implementation)
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
                link_id = max([l[0] for l in base.get('links', [])] or [0]) + 1
                prev_outputs = {}
                
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
                    
                    # Update combined I/O
                    for uid, in_info in info['inputs'].items():
                        if uid not in prev_outputs:
                            combined_inputs[uid] = {
                                'node_id': old_to_new_id[in_info['node_id']], 
                                'tags': in_info['tags'], 
                                'type': in_info['type']
                            }
                    
                    for uid, out_info in info['outputs'].items():
                        combined_outputs[uid] = {
                            'node_id': old_to_new_id[out_info['node_id']], 
                            'tags': out_info['tags'], 
                            'type': out_info['type']
                        }
                        prev_outputs[uid] = {
                            'node_id': old_to_new_id[out_info['node_id']], 
                            'type': out_info['type']
                        }
                
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

                if not self.validate_workflow(stitched):
                    raise ValueError("Stitched workflow validation failed")

                return {
                    'stitched_workflow': stitched, 
                    'inputs': combined_inputs, 
                    'outputs': combined_outputs, 
                    'execution_order': global_order
                }
                
            except nx.NetworkXError as e:
                self.log_message(f"Graph error on attempt {attempt+1}: {str(e)}", "error")
                if attempt == retries - 1:
                    raise
            except Exception as e:
                self.log_message(f"Error in stitch_workflows: {str(e)}", "error")
                raise
        
        raise ValueError("Max retries exceeded for stitching")

    async def run_sequential(self, workflow_paths: List[str], inputs: Dict[str, Any], 
                           iterations: int = 1, condition_port: Optional[str] = None, 
                           use_ram: bool = True, persist_prefix: Optional[str] = None, 
                           temp_dir: str = None, server_url: str = 'http://127.0.0.1:8188') -> Dict[str, Any]:
        """Execute workflows sequentially with data passing between iterations."""
        if not workflow_paths:
            self.log_message("No workflows provided", "error")
            raise ValueError("No workflows provided")
        
        self.log_message(f'Starting run_sequential with {len(workflow_paths)} workflows', 'info')
        
        # Register the DiscomfortDataLoader node
        if 'DiscomfortDataLoader' not in NODE_CLASS_MAPPINGS:
            from .nodes_internal import DiscomfortDataLoader
            NODE_CLASS_MAPPINGS['DiscomfortDataLoader'] = DiscomfortDataLoader
            self.log_message("Registered DiscomfortDataLoader node", "debug")
        
        # Set up storage
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
        
        # Clear any existing memory data
        clear_all_memory_data()
        
        intermediates = {}
        loop_inputs = inputs.copy()
        final_outputs = {}
        
        try:
            for iter_num in range(iterations):
                self.log_message(f"Starting iteration {iter_num + 1}", "info")
                extracted = {}
                
                for path in workflow_paths:
                    # Load original workflow
                    with open(path, 'r') as f:
                        original_workflow = json.load(f)
                    
                    info = self.discover_ports(path)
                    
                    # Get a clean workflow with reroutes removed
                    clean_workflow = self._get_workflow_with_reroutes_removed(original_workflow)
                    
                    # Inject data loaders for INPUT ports
                    modified_workflow = self._inject_data_loaders(
                        clean_workflow, loop_inputs, use_ram, storage_dir
                    )
                    
                    # Build prompt from modified workflow
                    prompt = self._build_prompt_from_workflow(modified_workflow)
                    self.log_message("Successfully built prompt from modified workflow.", "debug")
                    
                    # Generate IDs
                    unique_id = f'seq_{iter_num}_{os.path.basename(path)}_{time.time()}'
                    prompt_id = str(uuid.uuid4())
                    
                    # Validate the prompt
                    valid, errors, outputs_to_execute, node_errors = await execution.validate_prompt(prompt_id, prompt)
                    if not valid:
                        self.log_message(f"Validation failed with errors: {errors}", "error")
                        
                        # Log details about failing nodes
                        for node_id, error_details in node_errors.items():
                            node_info = prompt.get(int(node_id), {})
                            self.log_message(f"Node {node_id} ({node_info.get('class_type', 'unknown')}): {error_details}", "error")
                        
                        raise ValueError(f"Invalid prompt: {errors}")
                    
                    # Execute the workflow
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
                            raise TimeoutError(f"Timeout waiting for {path} in iteration {iter_num + 1}")
                        
                        # Log status periodically
                        if current_time - last_status_check > 5:
                            self.log_message(f"Still waiting for execution... ({elapsed:.1f}s elapsed)", "debug")
                            last_status_check = current_time
                        
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
                                    raise RuntimeError(error_msg)
                    
                    # Extract outputs
                    execution_outputs = execution_result.get('outputs', {})
                    
                    # Extract outputs from DiscomfortPort OUTPUT nodes
                    for uid, port_info in info['outputs'].items():
                        node_id_int = port_info['node_id']
                        node_id_str = str(node_id_int)
                        
                        # Check if the DiscomfortPort node has output
                        if node_id_str in execution_outputs:
                            node_output = execution_outputs[node_id_str]
                            if isinstance(node_output, dict) and 'output' in node_output:
                                # This is from a DiscomfortPort node that collected data
                                output_data = node_output['output']
                                if output_data is not None:
                                    # Store for next iteration
                                    if use_ram:
                                        intermediates[uid] = output_data
                                    else:
                                        serialized = self.serialize(output_data, use_ram)
                                        file_path = os.path.join(storage_dir, f"output_{uid}_{iter_num}.json")
                                        with open(file_path, 'w') as f:
                                            json.dump(serialized, f)
                                        intermediates[uid] = file_path
                                    
                                    extracted[uid] = intermediates[uid]
                                    self.log_message(f"Extracted output for port {uid} from node {node_id_int}", "debug")
                                else:
                                    self.log_message(f"No output data for port {uid} (node {node_id_int})", "warning")
                        else:
                            self.log_message(f"No output found for port {uid} (node {node_id_int}) in execution outputs", "warning")
                    
                    # Clean up memory data for this workflow
                    for uid in loop_inputs:
                        if uid in info['inputs']:
                            storage_key = f"{uid}_{time.time()}_{uuid.uuid4().hex[:8]}"
                            clear_data(storage_key, "memory")
                
                # Update loop inputs with extracted outputs
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
                        cond_value = cond_data
                    
                    if not bool(cond_value):
                        self.log_message(f"Condition {condition_port} evaluated False; stopping loop", "info")
                        break
        
        except Exception as e:
            self.log_message(f"Error in run_sequential: {str(e)}", "error")
            raise
        finally:
            # Cleanup
            if use_temp and storage_dir:
                shutil.rmtree(storage_dir)
                self.log_message(f"Cleaned up temp dir: {storage_dir}", "info")
            
            # Clear memory data
            clear_all_memory_data()
        
        # Return deserialized finals
        deserialized_outputs = {}
        for uid, data in final_outputs.items():
            if use_ram:
                deserialized_outputs[uid] = data
            elif isinstance(data, str):
                with open(data, 'r') as f:
                    serialized = json.load(f)
                deserialized_outputs[uid] = self.deserialize(serialized)
            else:
                deserialized_outputs[uid] = data
        
        return deserialized_outputs


# CLI interface remains the same
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
                    tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
                    inputs_dict[key] = tensor
        except json.JSONDecodeError:
            print("Invalid --inputs JSON; using empty dict.")
            inputs_dict = {}
        
        result = asyncio.run(tools.run_sequential(args.workflows, inputs_dict, args.iterations, args.condition_port, args.use_ram, args.persist_prefix, args.server_url))
        print(f'Final Outputs: {result}')
        
        # Postprocess: Save tensor outputs as images (CLI hack for testing)
        for uid, data in result.items():
            if isinstance(data, torch.Tensor) and data.dim() == 4:
                img_array = (data.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                img = Image.fromarray(img_array)
                save_path = f'output_{uid}.png'
                img.save(save_path)
                print(f'Saved output image for {uid} to {save_path}')