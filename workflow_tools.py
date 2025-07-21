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

# Import the WorkflowContext for high-performance, run-specific I/O
from .workflow_context import WorkflowContext


class DiscomfortWorkflowTools:
    """
    The engine for Discomfort, providing core functionality for port discovery,
    workflow stitching, and robust, stateful execution of iterative workflows.
    """
    def __init__(self):
        """Initializes the DiscomfortWorkflowTools and its dedicated logger."""
        self.logger = self._get_logger()

    def _get_logger(self):
        """
        Sets up and returns a dedicated logger for this class to ensure that
        log messages are namespaced and formatted consistently.
        """
        logger = logging.getLogger(f"DiscomfortWorkflowTools_{id(self)}")
        logger.propagate = False
        if not logger.hasHandlers():
            logger.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - [DiscomfortWorkflowTools] %(levelname)s - %(message)s')
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        return logger

    def _log_message(self, message: str, level: str = "info"):
        """Centralized logging method for the tools."""
        log_func = getattr(self.logger, level.lower(), self.logger.info)
        log_func(message)
        
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

    def validate_workflow(self, workflow: Dict[str, Any]) -> bool:
        """Validate workflow structure."""
        required_keys = ['nodes', 'links']
        for key in required_keys:
            if key not in workflow:
                self._log_message(f"Missing required key '{key}' in workflow", "error")
                return False
        # Check link integrity
        node_ids = {n['id'] for n in workflow['nodes']}
        for link in workflow['links']:
            if len(link) != 6:
                self._log_message("Invalid link format", "error")
                return False
            _, src, _, tgt, _, _ = link
            if src not in node_ids or tgt not in node_ids:
                self._log_message(f"Link references invalid node: {src} -> {tgt}", "error")
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
                    # Ensure widgets_values exists and has at least one element
                    wv = node.get('widgets_values', [])
                    if not wv or not wv[0]:
                        self._log_message(f"Node {node_id} is a DiscomfortPort but has no unique_id. Skipping.", "warning")
                        continue
                    unique_id = wv[0]
                    tags = wv[1].split(',') if len(wv) > 1 and wv[1] else []
                    
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
                    else: # Unconnected
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
                _, source_id, _, target_id, _, _ = link
                graph.add_edge(source_id, target_id)
            
            try:
                execution_order = list(nx.topological_sort(graph))
            except nx.NetworkXUnfeasible:
                raise ValueError('Workflow contains cycles; cannot perform topological sort.')
            
            return {
                'inputs': inputs,
                'outputs': outputs,
                'passthrus': passthrus,
                'execution_order': execution_order,
                'nodes': nodes,
                'links': links,
                'unique_id_to_node': unique_id_to_node
            }
        except Exception as e:
            self._log_message(f"Error in discover_ports for '{workflow_path}': {str(e)}", "error")
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
                self._log_message(f"Graph error on attempt {attempt+1}: {str(e)}", "error")
                if attempt == retries - 1:
                    raise
            except Exception as e:
                self._log_message(f"Error in stitch_workflows: {str(e)}", "error")
                raise
        
        raise ValueError("Max retries exceeded for stitching")

    def _prepare_prompt_for_contextual_run(self, prompt: Dict[str, Any], port_info: Dict[str, Any], context: WorkflowContext) -> Dict[str, Any]:
        """
        Modifies a prompt to work with WorkflowContext.
        1. Replaces INPUT DiscomfortPorts with DiscomfortDataLoaders.
        2. Injects the context's run_id into all DiscomfortPorts and DataLoaders.
        """
        self._log_message("Preparing prompt for contextual run...", "debug")
        modified_prompt = copy.deepcopy(prompt)
        run_id = context.run_id
        
        # Get discovered port information
        inputs_info = port_info.get('inputs', {})
        outputs_info = port_info.get('outputs', {})
        
        # Process INPUT ports: Replace with a DiscomfortDataLoader
        for unique_id, in_info in inputs_info.items():
            node_id_str = str(in_info['node_id'])
            if node_id_str not in modified_prompt:
                self._log_message(f"Node {node_id_str} for INPUT port '{unique_id}' not in prompt. Skipping.", "warning")
                continue

            self._log_message(f"Replacing INPUT port '{unique_id}' (node {node_id_str}) with a DataLoader.", "debug")
            modified_prompt[node_id_str] = {
                "inputs": {
                    "run_id": run_id,
                    "unique_id": unique_id,
                },
                "class_type": "DiscomfortDataLoader"
            }

        # Process ALL ports (Inputs, Outputs, Passthru) to inject run_id and set output flag
        all_port_nodes = {**inputs_info, **outputs_info, **port_info.get('passthrus', {})}
        for unique_id, p_info in all_port_nodes.items():
            node_id_str = str(p_info['node_id'])
            
            # Skip nodes that were already replaced by a DataLoader
            if node_id_str not in modified_prompt or modified_prompt[node_id_str]['class_type'] != 'DiscomfortPort':
                continue

            self._log_message(f"Injecting run_id '{run_id}' into port '{unique_id}' (node {node_id_str}).", "debug")
            if 'inputs' not in modified_prompt[node_id_str]:
                modified_prompt[node_id_str]['inputs'] = {}
            modified_prompt[node_id_str]['inputs']['run_id'] = run_id
            
            # If this is an OUTPUT port, flag it so it knows to save its data
            if unique_id in outputs_info:
                modified_prompt[node_id_str]['inputs']['is_output'] = True
                self._log_message(f"Marking port '{unique_id}' as OUTPUT.", "debug")
        
        return modified_prompt

    async def _run_sequential(self, workflow_paths: List[str], inputs: Dict[str, Any], 
                        iterations: int = 1, condition_port: Optional[str] = None, 
                        use_ram: bool = True, persist_prefix: Optional[str] = None, 
                        temp_dir: str = None, server_url: str = 'http://127.0.0.1:8188', connector: ComfyConnector = None) -> Dict[str, Any]:
        """
        (Coroutine, Internal) Execute workflows sequentially with high-performance data passing using WorkflowContext.
        This method orchestrates the entire run, from context creation to data I/O and execution.
        This method is called by run_sequential and should not be called directly.
        """
        # --- Pre-computation Step ---
        # Discover ports and load workflows once to avoid redundant I/O in the loop.
        all_ports_info = {path: self.discover_ports(path) for path in workflow_paths}
        all_original_workflows = {}
        for path in workflow_paths:
            with open(path, 'r') as f:
                all_original_workflows[path] = json.load(f)

        # Aggregate all unique output port IDs from all workflows
        all_output_unique_ids = set()
        for port_info in all_ports_info.values():
            all_output_unique_ids.update(port_info['outputs'].keys())
        self._log_message(f"Aggregated output ports for this run: {list(all_output_unique_ids)}", "debug")

        # --- Context Setup ---
        # The WorkflowContext manages all data I/O for this run. Using a `with` statement
        # ensures its resources (shared memory, temp files) are automatically cleaned up.
        try:
            with WorkflowContext() as context:
                self._log_message(f"Created WorkflowContext for this run with ID: {context.run_id}", "info")
                
                # Save all initial inputs to the context before the loop starts.
                self._log_message(f"Saving initial inputs to context: {list(inputs.keys())}", "debug")
                for unique_id, data in inputs.items():
                    context.save(unique_id, data, use_ram=use_ram)
                    self._log_message(f"Initial inputs saved to context: {unique_id}", "debug")
            
                # --- EXECUTION LOOP ---
                final_outputs = {}
                loop_condition_met = True

                for iter_num in range(iterations):
                    if not loop_condition_met:
                        break # Exit loop if condition was not met in the previous iteration

                    self._log_message(f"--- Starting Iteration {iter_num + 1}/{iterations} ---", "info")
                    
                    for path_idx, path in enumerate(workflow_paths):
                        self._log_message(f"Processing workflow {path_idx + 1}/{len(workflow_paths)}: '{os.path.basename(path)}'", "info")

                        # Use pre-loaded workflow and port info
                        original_workflow = all_original_workflows[path]
                        port_info = all_ports_info[path]
                        self._log_message(f"Discovered INPUT ports: {list(port_info['inputs'].keys())}", "debug")
                        self._log_message(f"Discovered OUTPUT ports: {list(port_info['outputs'].keys())}", "debug")

                        # Convert the workflow to an API-ready prompt
                        self._log_message("Converting workflow to prompt JSON...", "debug")
                        prompt = await connector._get_prompt_from_workflow(original_workflow)
                        
                        # Prepare the prompt for this run by injecting the context_id and replacing input ports
                        modified_prompt = self._prepare_prompt_for_contextual_run(prompt, port_info, context)

                        # *** EXECUTION STEP ***
                        self._log_message(f"Executing modified prompt for workflow '{os.path.basename(path)}'.", "info")
                        execution_result = await connector.run_workflow(modified_prompt, use_workflow_json=False)
                        
                        if not execution_result:
                            self._log_message(f"Workflow '{os.path.basename(path)}' execution failed to produce a result. Aborting run.", "error")
                            raise RuntimeError(f"Workflow '{os.path.basename(path)}' execution failed.")

                    # --- Post-Iteration Data Handling ---
                    # After each full iteration, load all possible outputs from the context.
                    # This ensures the state is up-to-date for condition checks and the final return value.
                    self._log_message("Loading all iteration outputs from context...", "debug")
                    current_iter_outputs = {}
                    for uid in all_output_unique_ids:
                        try:
                            # Load data from context, the single source of truth for I/O.
                            data = context.load(uid)
                            current_iter_outputs[uid] = data
                            self._log_message(f"Successfully loaded output for port '{uid}'.", "debug")
                        except KeyError:
                            # This is not an error; an output port may not have been executed.
                            self._log_message(f"No output found in context for port '{uid}' in this iteration.", "debug")
                    
                    final_outputs.update(current_iter_outputs)
                    
                    # Check condition port if specified
                    if condition_port and condition_port in final_outputs:
                        cond_data = final_outputs[condition_port]
                        loop_condition_met = bool(cond_data) if not isinstance(cond_data, (int, float, str)) else bool(cond_data)
                        if not loop_condition_met:
                            self._log_message(f"Condition port '{condition_port}' evaluated to False. Stopping loop.", "info")
                
                self._log_message(f"Usage report after iteration {iter_num+1}: {context.get_usage()}", "debug")
                return final_outputs
        except Exception as e:
            self._log_message(f"An error occurred during run_sequential: {str(e)}", "error")
            import traceback
            traceback.print_exc()
            raise
        finally:
            self._log_message("run_sequential finished. WorkflowContext will now clean up resources.", "info")
    
    async def run_sequential(self, workflow_paths: List[str], inputs: Dict[str, Any], 
                        iterations: int = 1, condition_port: Optional[str] = None, 
                        use_ram: bool = True, persist_prefix: Optional[str] = None, 
                        temp_dir: str = None, server_url: str = 'http://127.0.0.1:8188') -> Dict[str, Any]:
        """
        (Coroutine) Execute workflows sequentially with high-performance data passing using WorkflowContext.
        This method orchestrates the entire run, from context creation to data I/O and execution, using _run_sequential as the main execution loop.
        """
        if not workflow_paths:
            self._log_message("No workflows provided to run_sequential", "error")
            raise ValueError("No workflows provided")
        self._log_message(f'Starting run_sequential for {len(workflow_paths)} workflow(s) over {iterations} iteration(s).', 'info')
        try:
            connector = await ComfyConnector.create() # We create a new connector here to avoid re-initialization inside the _run_sequential method.
            while connector._state != "ready":
                self._log_message(f"Waiting for connector to be fully initialized... (State: {connector._state})", "info")
                await asyncio.sleep(0.5)
            return await self._run_sequential(workflow_paths, inputs, iterations, condition_port, use_ram, persist_prefix, temp_dir, server_url, connector)
        except Exception as e:
            self._log_message(f"An error occurred during run_sequential: {str(e)}", "error")
            import traceback
            traceback.print_exc()
            raise