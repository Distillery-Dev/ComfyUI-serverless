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
import asyncio
from typing import Dict, List, Any, Optional
import argparse
import uuid

import numpy as np
from PIL import Image
import logging

# This try/except block is for when the server isn't running, which is fine.
try:
    import server
except ModuleNotFoundError:
    server = None

# Import the comfy_serverless module for nested ComfyUI execution
from .comfy_serverless import ComfyConnector

# Import the internal data storage functions
from .nodes_internal import store_data, clear_data, clear_all_memory_data


class DiscomfortWorkflowTools:
    def log_message(self, message: str, level: str = "info"):
        """Centralized logging method for Discomfort."""
        # Set up basic logging if not already configured
        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
        
        msg = f"[Discomfort] {level.upper()}: {message}"
        levels = {'debug': logging.DEBUG, 'info': logging.INFO, 'warning': logging.WARNING, 'error': logging.ERROR}
        level_val = levels.get(level.lower(), logging.INFO)
        
        # Use a dedicated logger for Discomfort to avoid interfering with ComfyUI's root logger
        discomfort_logger = logging.getLogger("Discomfort")
        discomfort_logger.setLevel(logging.DEBUG) # Ensure all levels can be processed
        
        # Prevent duplicate handlers
        if not discomfort_logger.handlers:
            # Add a handler if none exist, e.g., to print to console
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - [Discomfort] %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            discomfort_logger.addHandler(ch)
            discomfort_logger.propagate = False # Stop messages from going to the root logger

        # Get the actual logging function and call it
        log_func = getattr(discomfort_logger, level.lower(), discomfort_logger.info)
        log_func(message) # Pass the original message without the prefix, as the formatter handles it

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

    async def _inject_data_loaders(self, workflow: Dict[str, Any], port_data: Dict[str, Any], 
                                   storage_dir: str, is_ephemeral: bool, connector: ComfyConnector) -> Dict[str, Any]:
        """
        (Coroutine) Replace DiscomfortPort INPUT nodes with DiscomfortDataLoader nodes by modifying the workflow JSON.
        This is an async function because it performs file I/O for data serialization and upload.
        """
        self.log_message("Injecting data loaders into workflow...", "debug")
        modified_workflow = copy.deepcopy(workflow)
        nodes_to_remove = []
        nodes_to_add = []
        
        # Process each node in the workflow
        for node_idx, node in enumerate(modified_workflow['nodes']):
            if node.get('type') == 'DiscomfortPort':
                # Get the unique_id from widgets_values
                wv = node.get('widgets_values', [])
                unique_id = wv[0] if wv else ''
                
                # An INPUT port has no incoming connections to its first input slot
                node_id = node['id']
                has_incoming = any(link[3] == node_id and link[4] == 0 for link in modified_workflow.get('links', []))
                
                if not has_incoming and unique_id in port_data:
                    self.log_message(f"Found INPUT port '{unique_id}' to replace.", "debug")
                    
                    # Serialize the data and save to a JSON file in the storage directory
                    data = port_data[unique_id]
                    serialized = self.serialize(data)
                    storage_key = f"{uuid.uuid4()}.json"  # Unique filename for the serialized data
                    file_path = os.path.join(storage_dir, storage_key)
                    with open(file_path, 'w') as f:
                        json.dump(serialized, f)
                    
                    # Upload the serialized file to the nested ComfyUI's temp directory.
                    # This is now a direct await, assuming connector.upload_data is a coroutine.
                    dest_path = connector.upload_data(
                        source_path=file_path, 
                        folder_type='temp', 
                        subfolder=None, 
                        overwrite=True, 
                        is_ephemeral=is_ephemeral
                    )
                    
                    # Create the DiscomfortDataLoader node to replace the DiscomfortPort
                    new_node = {
                        'id': node_id,  # Keep the same ID to preserve connections
                        'type': 'DiscomfortDataLoader',
                        'pos': node.get('pos', [0, 0]),
                        'size': node.get('size', [200, 100]),
                        'flags': node.get('flags', {}),
                        'order': node.get('order', 0),
                        'mode': node.get('mode', 0),
                        'inputs': [],  # DataLoader has no inputs
                        'outputs': node.get('outputs', []),  # Keep the same outputs
                        'properties': {},
                        'widgets_values': [dest_path, "disk", "ANY"]  # Use absolute dest_path, disk type, ANY expected (general-purpose)
                    }
                    
                    # Mark the old node for removal and the new one for addition
                    nodes_to_remove.append(node_idx)
                    nodes_to_add.append((node_idx, new_node))
        
        # Apply the changes to the workflow
        if nodes_to_add:
            # Remove nodes in reverse order to avoid index issues
            for idx in sorted(nodes_to_remove, reverse=True):
                del modified_workflow['nodes'][idx]
            
            # Insert new nodes at the original positions
            for idx, new_node in sorted(nodes_to_add):
                modified_workflow['nodes'].insert(idx, new_node)
            self.log_message(f"Replaced {len(nodes_to_add)} DiscomfortPort nodes with DataLoaders.", "info")
        else:
            self.log_message("No INPUT ports found to replace.", "debug")

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
        """A robust method to remove reroute nodes and correctly rewire all connections."""
        
        # --- PASS 1: MAP REROUTE FLOWS ---
        # This map will store the final destination of any reroute chain.
        # e.g., {reroute_A: (source_node, source_slot)}
        reroute_source_map = {}
        
        def find_true_source(node_id, slot):
            # Recursively trace back to the original, non-reroute source node
            if nodes.get(node_id, {}).get('type') == 'Reroute':
                # Find the link that feeds *into* this reroute node
                for link in links:
                    if link[3] == node_id: # target_id matches
                        # Continue tracing from this link's source
                        return find_true_source(link[1], link[2])
            # Base case: we've found a non-reroute node
            return node_id, slot

        # --- PASS 2: REWIRE LINKS ---
        new_links = []
        new_nodes = {nid: n for nid, n in nodes.items() if n['type'] != 'Reroute'}
        link_id_map = {}

        for link in links:
            original_link_id = link[0]
            source_id, source_slot = link[1], link[2]
            target_id, target_slot = link[3], link[4]
            link_type = link[5]

            # If the source of this link is a reroute, find its true origin
            true_source_id, true_source_slot = find_true_source(source_id, source_slot)
            
            # If the target of this link is a reroute, we don't need to do anything special here,
            # because the next link in the chain will handle it by looking at its own source.
            
            # Only add links that connect to non-reroute nodes.
            if nodes.get(true_source_id, {}).get('type') != 'Reroute' and \
               nodes.get(target_id, {}).get('type') != 'Reroute':
                
                # Create the new, direct link
                new_link = [original_link_id, true_source_id, true_source_slot, target_id, target_slot, link_type]
                new_links.append(new_link)
                link_id_map[original_link_id] = original_link_id
        
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
        """(Coroutine) Execute workflows sequentially with data passing between iterations using a nested ComfyUI server."""
        # Defer imports to prevent circular dependencies at the module level
        import nodes

        if not workflow_paths:
            self.log_message("No workflows provided to run_sequential", "error")
            raise ValueError("No workflows provided")
        
        self.log_message(f'Starting run_sequential for {len(workflow_paths)} workflow(s) over {iterations} iteration(s).', 'info')
        
        # Ensure the DiscomfortDataLoader node is registered in the running ComfyUI instance
        if 'DiscomfortDataLoader' not in nodes.NODE_CLASS_MAPPINGS:
            from .nodes_internal import DiscomfortDataLoader
            nodes.NODE_CLASS_MAPPINGS['DiscomfortDataLoader'] = DiscomfortDataLoader
            self.log_message("Registered DiscomfortDataLoader node with ComfyUI.", "debug")
        
        # --- STORAGE SETUP ---
        if use_ram:
            self.log_message("use_ram=True is not supported with nested ComfyUI execution; forcing disk storage.", "warning")
        
        storage_dir = None
        is_ephemeral = True
        if persist_prefix:
            storage_dir = persist_prefix
            os.makedirs(storage_dir, exist_ok=True)
            is_ephemeral = False
            self.log_message(f"Using persistent disk storage at: {storage_dir}", "info")
        else:
            storage_dir = temp_dir or tempfile.mkdtemp(prefix="discomfort_run_")
            self.log_message(f"Using temporary disk storage at: {storage_dir}", "info")
        
        clear_all_memory_data()
        self.log_message("Cleared all in-memory data from previous runs.", "debug")
        
        # Instantiate the nested ComfyUI connector
        # Assuming ComfyConnector's __init__ is synchronous. If it's async, it needs `await`.
        connector = ComfyConnector()
        
        # --- EXECUTION LOOP ---
        loop_inputs = inputs.copy()
        final_outputs = {}
        
        try:
            for iter_num in range(iterations):
                self.log_message(f"--- Starting Iteration {iter_num + 1}/{iterations} ---", "info")
                extracted_for_next_iter = {}
                
                for path_idx, path in enumerate(workflow_paths):
                    self.log_message(f"Processing workflow {path_idx + 1}/{len(workflow_paths)}: '{os.path.basename(path)}'", "info")

                    with open(path, 'r') as f:
                        original_workflow = json.load(f)
                    
                    info = self.discover_ports(path)
                    self.log_message(f"Discovered INPUT ports: {list(info['inputs'].keys())}", "debug")
                    self.log_message(f"Discovered OUTPUT ports: {list(info['outputs'].keys())}", "debug")

                    clean_workflow = self._get_workflow_with_reroutes_removed(original_workflow)
                    
                    modified_workflow = await self._inject_data_loaders(
                        clean_workflow, loop_inputs, storage_dir, is_ephemeral, connector
                    )
                    
                    # Await the coroutine directly. Do not wrap in asyncio.to_thread.
                    prompt = await connector._get_prompt_from_workflow(modified_workflow)
                    
                    self.log_message(f"Generated Prompt ID: {str(uuid.uuid4())}", "debug")

                    # *** EXECUTION STEP ***
                    self.log_message(f"Executing prompt on nested ComfyUI server.", "info")
                    
                    # Await the coroutine directly.
                    execution_result = await connector.run_workflow(prompt, use_workflow_json=False)
                    
                    # --- Output Extraction ---
                    # Now execution_result is the actual dictionary result, not a coroutine.
                    execution_outputs = execution_result.get('outputs', {})
                    prompt_id = execution_result.get('prompt_id', 'N/A')
                    self.log_message(f"Extracting outputs for prompt {prompt_id}...", "debug")

                    for uid, port_info in info['outputs'].items():
                        node_id_str = str(port_info['node_id'])
                        
                        if node_id_str in execution_outputs:
                            node_output_data = execution_outputs[node_id_str]
                            path = node_output_data.get('outputs', {}).get(0, None)
                            if path and os.path.exists(path):
                                with open(path, 'r') as f:
                                    serialized = json.load(f)
                                extracted_data = self.deserialize(serialized)
                                if extracted_data is not None:
                                    extracted_for_next_iter[uid] = extracted_data
                                    self.log_message(f"Extracted output for port '{uid}' from node {node_id_str}.", "debug")
                                if is_ephemeral:
                                    os.remove(path)
                            else:
                                self.log_message(f"No valid output path found for port '{uid}' (node {node_id_str}).", "warning")
                        else:
                            self.log_message(f"No output found in execution history for port '{uid}' (node {node_id_str}).", "warning")
                
                loop_inputs.update(extracted_for_next_iter)
                final_outputs.update(extracted_for_next_iter)
                
                if condition_port and condition_port in final_outputs:
                    cond_data = final_outputs[condition_port]
                    cond_value = bool(cond_data) if not isinstance(cond_data, (int, float, str)) else bool(cond_data)
                    if not cond_value:
                        self.log_message(f"Condition port '{condition_port}' evaluated to False. Stopping loop.", "info")
                        break
        
        except Exception as e:
            self.log_message(f"An error occurred during run_sequential: {str(e)}", "error")
            import traceback
            traceback.print_exc()
            raise
        finally:
            # --- Cleanup ---
            # Await the coroutine directly.
            await connector.kill_api()

            if is_ephemeral and storage_dir and os.path.exists(storage_dir):
                shutil.rmtree(storage_dir)
                self.log_message(f"Cleaned up temporary directory: {storage_dir}", "info")
            clear_all_memory_data()
            self.log_message("Final cleanup of all in-memory data.", "debug")
        
        self.log_message("run_sequential finished.", "info")
        return final_outputs


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
        
        # Setup asyncio event loop to run the async function from a sync context
        result = asyncio.run(tools.run_sequential(
            args.workflows, 
            inputs_dict, 
            args.iterations, 
            args.condition_port, 
            args.use_ram, 
            args.persist_prefix,
            server_url=args.server_url
        ))
        print(f'Final Outputs: {result}')
        
        # Postprocess: Save tensor outputs as images (CLI hack for testing)
        for uid, data in result.items():
            if isinstance(data, torch.Tensor) and data.dim() == 4:
                img_array = (data.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                img = Image.fromarray(img_array)
                save_path = f'output_{uid}.png'
                img.save(save_path)
                print(f'Saved output image for {uid} to {save_path}')