import json
import networkx as nx
import os
import copy
import asyncio
import tempfile
from typing import Dict, List, Any, Optional
# Import the comfy_serverless module for nested ComfyUI execution
from .comfy_serverless import ComfyConnector
# Import the WorkflowContext for high-performance, run-specific I/O
from .workflow_context import WorkflowContext


class WorkflowTools:
    """
    The engine for Discomfort, providing core functionality for port discovery,
    workflow stitching, and robust, stateful execution of iterative workflows.
    """
    def __init__(self):
        """Initializes the WorkflowTools and its dedicated logger."""
        self.logger = self._get_logger()
        # NEW: Load pass-by-reference rules on initialization
        self.pass_by_rules = self._load_pass_by_rules()

    def _get_logger(self):
        """
        Sets up and returns a dedicated logger for this class to ensure that
        log messages are namespaced and formatted consistently, including the method name that calls _log_message.
        """
        import logging
        import inspect

        class CustomFormatter(logging.Formatter):
            def format(self, record):
                frame = inspect.currentframe().f_back  # Start from caller of format (emit)
                while frame and frame.f_code.co_name != '_log_message':
                    frame = frame.f_back
                if frame:
                    # One more step back to get the caller of _log_message
                    caller_frame = frame.f_back
                    record.caller_funcName = caller_frame.f_code.co_name if caller_frame else "<unknown>"
                else:
                    record.caller_funcName = "<unknown>"
                return super().format(record)

        logger = logging.getLogger(f"WorkflowTools_{id(self)}")
        logger.propagate = False
        if not logger.hasHandlers():
            logger.setLevel(logging.DEBUG)
            formatter = CustomFormatter('%(asctime)s - [WorkflowTools] (%(caller_funcName)s) %(levelname)s - %(message)s')
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        return logger
    
    # NEW: Method to load the pass_by_rules.json configuration.
    def _load_pass_by_rules(self) -> Dict[str, str]:
        """Loads the pass-by-reference rules from the JSON config file."""
        rules_path = os.path.join(os.path.dirname(__file__), 'pass_by_rules.json')
        try:
            with open(rules_path, 'r') as f:
                rules = json.load(f)
            self._log_message(f"Loaded pass-by-reference rules from {rules_path}", "info")
            return rules
        except FileNotFoundError:
            self._log_message(f"pass_by_rules.json not found at {rules_path}. All types will be passed by value.", "warning")
            return {}
        except json.JSONDecodeError:
            self._log_message(f"Error decoding pass_by_rules.json. All types will be passed by value.", "error")
            return {}

    def _log_message(self, message: str, level: str = "info"):
        """Centralized logging method for the handler."""
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

    def discover_port_nodes(self, workflow_path: str) -> Dict[str, Any]:
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
                        self._log_message(f"INPUT '{unique_id}' is of type '{inferred_type}'.", "debug")
                    if not outgoing:
                        outputs[unique_id] = port_info
                        self._log_message(f"OUTPUT '{unique_id}' is of type '{inferred_type}'.", "debug")
                    if incoming and outgoing:
                        passthrus[unique_id] = port_info
                        self._log_message(f"PASSTHRU '{unique_id}' is of type '{inferred_type}'.", "debug")
                    
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
            self._log_message(f"Error in discover_port_nodes for '{workflow_path}': {str(e)}", "error")
            raise

    # NEW: Implementation of the workflow pruning logic.
    def _prune_workflow_to_output(self, workflow: Dict[str, Any], target_output_unique_id: str) -> Dict[str, Any]:
        """
        Prunes a workflow to the minimal set of nodes required to generate a specific output port.
        """
        self._log_message(f"Pruning workflow to generate only output '{target_output_unique_id}'...", "debug")

        # 1. Discover ports to find the target node ID
        # We need the full workflow info, so we can't use a pre-cached version.
        # A bit inefficient, but necessary for correctness. We'll create a temporary file.
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".json") as temp_f:
            json.dump(workflow, temp_f)
            temp_path = temp_f.name
        
        try:
            port_info = self.discover_port_nodes(temp_path)
        finally:
            os.remove(temp_path)

        if target_output_unique_id not in port_info['outputs']:
            raise KeyError(f"Target output '{target_output_unique_id}' not found in the workflow's output ports.")
        
        target_node_id = port_info['outputs'][target_output_unique_id]['node_id']

        # 2. Build a graph and find all ancestors of the target node
        graph = nx.DiGraph()
        for node in workflow['nodes']:
            graph.add_node(node['id'])
        for link in workflow['links']:
            # link format: [id, source_node_id, source_slot, target_node_id, target_slot, type]
            graph.add_edge(link[1], link[3])
        
        # nx.ancestors finds all nodes that have a path to target_node_id
        required_node_ids = nx.ancestors(graph, target_node_id)
        required_node_ids.add(target_node_id) # The target node itself is also required

        self._log_message(f"Found {len(required_node_ids)} required nodes for '{target_output_unique_id}'.", "debug")

        # 3. Create the new, pruned workflow
        pruned_nodes = [node for node in workflow['nodes'] if node['id'] in required_node_ids]
        pruned_links = [link for link in workflow['links'] if link[1] in required_node_ids and link[3] in required_node_ids]

        pruned_workflow = copy.deepcopy(workflow) # Keep metadata
        pruned_workflow['nodes'] = pruned_nodes
        pruned_workflow['links'] = pruned_links

        self._log_message(f"Successfully pruned workflow for '{target_output_unique_id}'.", "info")
        return pruned_workflow

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
                    info = self.discover_port_nodes(path)
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

    def _prepare_prompt_for_contextual_run(self, prompt: Dict[str, Any], port_info: Dict[str, Any], context: WorkflowContext, uids_handled_by_stitch: set = None) -> Dict[str, Any]:
        """
        Modifies a prompt to work with WorkflowContext by performing class-swapping.
        This version uses the port's inherent type to determine swapping logic.
        """
        if uids_handled_by_stitch is None:
            uids_handled_by_stitch = set()

        self._log_message("Preparing prompt for contextual run via class-swapping...", "debug")
        modified_prompt = copy.deepcopy(prompt) # Deepcopy the original prompt to avoid modifying it.
        run_id = context.run_id # This is the identifier for the context to load/save data from.
        
        inputs_info = port_info.get('inputs', {}) # This is the information about the INPUT DiscomfortPorts.
        outputs_info = port_info.get('outputs', {}) # This is the information about the OUTPUT DiscomfortPorts.
        
        # --- Step 1: Process INPUT ports ---
        for unique_id, in_info in inputs_info.items():
            node_id_str = str(in_info['node_id'])
            
            # Sanity Check: if the node_id_str is not in the prompt, continue.
            if node_id_str not in modified_prompt: 
                continue

            # Avoid swapping any unique_ids already satisfied by stitching (ie, 'ref' unique_ids).
            if unique_id in uids_handled_by_stitch:
                self._log_message(f"Skipping INPUT port swap for '{unique_id}' (pass-by-reference).", "debug")
                continue
            # Sanity Check: if the port is a 'ref' type, it should have been stitched or is an error.
            port_type = out_info.get('type', 'ANY').upper()
            pass_by_method = self.pass_by_rules.get(port_type, 'val')
            if pass_by_method == 'ref':
                self._log_message(f"Skipping INPUT port swap for '{unique_id}' (pass-by-reference).", "debug")
                continue

            # Now, we swap the INPUT DiscomfortPort with a DiscomfortContextLoader using the appropriate run_id and unique_id.
            self._log_message(f"Swapping INPUT port '{unique_id}' (node {node_id_str}) with DiscomfortDataLoader.", "debug")
            modified_prompt[node_id_str] = {
                "inputs": {"run_id": run_id, "unique_id": unique_id},
                "class_type": "DiscomfortContextLoader"
            }

        # --- Step 2: Process OUTPUT ports ---
        for unique_id, out_info in outputs_info.items():
            node_id_str = str(out_info['node_id'])

            # Sanity Check: if the node_id_str is not in the prompt, continue.
            if node_id_str not in modified_prompt:
                continue

            # Avoid swapping any unique_ids already satisfied by stitching (ie, 'ref' unique_ids).
            if unique_id in uids_handled_by_stitch: 
                self._log_message(f"Skipping OUTPUT port swap for '{unique_id}' (pass-by-reference).", "debug")
                continue
            # Sanity Check: if the port is a 'ref' type, it should have been stitched or is an error.
            port_type = out_info.get('type', 'ANY').upper()
            pass_by_method = self.pass_by_rules.get(port_type, 'val')
            if pass_by_method == 'ref':
                self._log_message(f"Skipping OUTPUT port swap for '{unique_id}' (pass-by-reference).", "debug")
                continue

            # Now, we swap the OUTPUT DiscomfortPort with a DiscomfortContextSaver using the appropriate run_id and unique_id.
            self._log_message(f"Swapping OUTPUT port '{unique_id}' (node {node_id_str}) with DiscomfortContextSaver.", "debug")
            original_node = prompt[node_id_str] # This is the original node that we are swapping.
            data_input = original_node['inputs'].get('input_data') # This is the data that will be saved to the context.
            modified_prompt[node_id_str] = {
                "inputs": {
                "input_data": data_input, "unique_id": unique_id,"run_id": run_id},
                "class_type": "DiscomfortContextSaver"
            }
        
        # At this point, all the INPUT and OUTPUT DiscomfortPorts have been swapped with the appropriate nodes.
        return modified_prompt

    async def run_sequential(self, workflow_paths: List[str], inputs: Dict[str, Any], iterations: int = 1, use_ram: bool = True) -> Dict[str, Any]:
        
        # --- Pre-processing Step ---
        # Hygiene check: if no workflows were given, raise an error.
        if not workflow_paths:
            raise ValueError("No workflows provided")
        self._log_message(f'Starting run_sequential for {len(workflow_paths)} workflow(s) over {iterations} iteration(s).', 'info')
        
        # We will now load each workflow into memory and discover its DiscomfortPorts.
        all_ports_info = {}
        all_original_workflows = {}
        for path in workflow_paths: # Open each workflow and save it to a temporary file.
            with open(path, 'r') as f:
                wf_data = json.load(f)
                all_original_workflows[path] = wf_data
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".json") as temp_f:
                    json.dump(wf_data, temp_f)
                    temp_path = temp_f.name
                try:
                    all_ports_info[path] = self.discover_port_nodes(temp_path) # Discover the DiscomfortPorts in each workflow.
                finally:
                    os.remove(temp_path) # Remove the temporary file.

        # We will now create a map of all the inputs to be added to the context.
        all_inputs_map = {}
        for port_info in all_ports_info.values():
            for uid, info in port_info.get('inputs', {}).items():
                if uid not in all_inputs_map:
                    all_inputs_map[uid] = info

        # --- Context Setup ---
        try:
            with WorkflowContext() as context: # This line creates a new context for this run and ensures it is cleaned up in the end.
                connector = await ComfyConnector.create() # Starts or reconnects to the nested ComfyUI instance.
                while connector._state != "ready":
                    await asyncio.sleep(0.5)
                self._log_message(f"Created WorkflowContext for this run with ID: {context.run_id}", "info")
                
                # We will now add all the inputs to the context.
                for unique_id, data in inputs.items(): # Collect all the inputs to the context.
                    port_type = all_inputs_map.get(unique_id, {}).get('type', 'ANY').upper()
                    pass_by_method = self.pass_by_rules.get(port_type, 'val')
                    context.save(unique_id, data, use_ram=use_ram, pass_by=pass_by_method) # Add the input to the context together with its pass-by-method.
                    self._log_message(f"Initial input '{unique_id}' saved to context as type '{pass_by_method}'.", "debug")
                
                # --- EXECUTION LOOP ---
                final_outputs = {} # This will store the outputs of the final iteration.
                
                # If iterations > 1, we will iterate for all iterations.
                for iter_num in range(iterations): 
                    if iterations == 1: # If there is only one iteration, we will not log the iteration number.
                        self._log_message(f"--- Starting Sequential Run ---", "info")
                    else: # If there are multiple iterations, we will log the iteration number.
                        self._log_message(f"--- Starting Sequential Run - Iteration {iter_num + 1}/{iterations} ---", "info")
                    
                    # If there are chained workflows, we will iterate for all of them.
                    for path_idx, path in enumerate(workflow_paths): 
                        # Log the current workflow being processed.
                        if len(workflow_paths) > 1: # If there are multiple workflows, we will log the current workflow number.
                            self._log_message(f"Processing workflow {path_idx + 1}/{len(workflow_paths)}: '{os.path.basename(path)}'", "info")
                        else: # If there is only one workflow, we will not log the workflow number.
                            self._log_message(f"Processing workflow: '{os.path.basename(path)}'", "info")

                        # Load the original workflow and its port information.
                        original_workflow = all_original_workflows[path] # this is the original workflow JSON
                        port_info = all_ports_info[path] # this tells us which DiscomfortPorts are in the workflow
                        current_workflow = copy.deepcopy(original_workflow) # deepcopy the original workflow to avoid modifying it
                        ref_workflows_to_stitch = [] # this will store the workflows that lead to the 'ref' unique_ids
                        uids_handled_by_stitch = set() # this will store the 'ref' unique_ids that will be handled by stitching

                        # We now identify unique_ids that are pass-by-reference or pass-by-value.
                        # Pass-by-reference inputs are handled by stitching, whereas
                        # pass-by-value inputs are directly saved to/imported from the context. 
                        for uid, in_info in port_info['inputs'].items():
                            port_type = in_info.get('type', 'ANY').upper()
                            pass_by_method = self.pass_by_rules.get(port_type, 'val')
                            
                            # Stitching is only used if the port is a 'ref' type, ie pass-by-reference.
                            if pass_by_method == 'ref':
                                # Sanity Check: check if the needed reference exists in the context from a previous step.
                                if context.get_storage_info(uid):
                                    self._log_message(f"Found pass-by-reference input: '{uid}'. Preparing to stitch.", "info")
                                    minimal_workflow = context.load(uid) # This is the pruned workflow that leads to the 'ref' unique_id.
                                    ref_workflows_to_stitch.append(minimal_workflow) # These are the workflows that lead to the 'ref' unique_ids.
                                    uids_handled_by_stitch.add(uid) # This is a set of 'ref' unique_ids, which will be handled by stitching.
                                else:
                                    self._log_message(f"Input '{uid}' is 'ref' type but was not found in context. It will be swapped to a (likely failing) ContextLoader.", "warning")
                        
                        if ref_workflows_to_stitch: # If there are any 'ref' unique_ids to be handled by stitching, we will stitch the workflows.
                            self._log_message(f"Stitching {len(ref_workflows_to_stitch)} reference workflows...", "info")
                            # Create temporary files for stitching
                            temp_files = [] # This list will store the paths to the workflows that must be stitched.
                            # First, store the main workflow
                            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".json") as temp_f:
                                json.dump(current_workflow, temp_f)
                                temp_files.append(temp_f.name)
                            # Then, store the reference workflows
                            for ref_wf in ref_workflows_to_stitch:
                                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".json") as temp_f:
                                    json.dump(ref_wf, temp_f)
                                    temp_files.append(temp_f.name)
                            
                            try:
                                # Stitch the reference workflows *before* the main workflow is executed.
                                # This ensures that the workflow to be executed gets to the 'ref' unique_ids it needs.
                                stitch_result = self.stitch_workflows(temp_files[1:] + [temp_files[0]]) # stitch the workflows
                                current_workflow = stitch_result['stitched_workflow'] # this is the stitched workflow
                                # The port info is now stale, so we must rediscover it for the new stitched graph
                                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".json") as temp_f:
                                    json.dump(current_workflow, temp_f)
                                    new_path = temp_f.name
                                port_info = self.discover_port_nodes(new_path)
                                os.remove(new_path)
                                self._log_message("Stitching complete. Port info has been updated.", "info")
                            finally:
                                for f in temp_files:
                                    os.remove(f)

                        # Convert the (potentially stitched) workflow to an API-ready prompt
                        self._log_message("Converting workflow to prompt JSON...", "debug")
                        prompt = await connector.get_prompt_from_workflow(current_workflow)
                        
                        # At this point, all the 'ref' unique_ids have been stitched into the workflow.
                        # We now prepare the prompt for this run by replacing the INPUT and OUTPUT DiscomfortPorts with the appropriate nodes.
                        modified_prompt = self._prepare_prompt_for_contextual_run(prompt, port_info, context, uids_handled_by_stitch)

                        # *** EXECUTION STEP ***
                        self._log_message(f"Executing modified prompt for workflow '{os.path.basename(path)}'.", "info")
                        execution_result = await connector.run_workflow(modified_prompt, use_workflow_json=False)
                        
                        if not execution_result:
                            self._log_message(f"Workflow '{os.path.basename(path)}' execution failed to produce a result. Aborting run.", "error")
                            raise RuntimeError(f"Workflow '{os.path.basename(path)}' execution failed.")

                        # --- CORRECTED POST-EXECUTION OUTPUT HANDLING ---
                        self._log_message(f"Processing outputs from '{os.path.basename(path)}'...", "debug")
                        # Use the port_info of the workflow that just ran.
                        for uid, out_info in port_info['outputs'].items():
                            port_type = out_info.get('type', 'ANY').upper()
                            pass_by_method = self.pass_by_rules.get(port_type, 'val')

                            if pass_by_method == 'ref':
                                self._log_message(f"Output '{uid}' is pass-by-reference. Pruning workflow.", "info")
                                pruned_wf = self._prune_workflow_to_output(original_workflow, uid)
                                context.save(uid, pruned_wf, use_ram=use_ram, pass_by='ref')
                                final_outputs[uid] = pruned_wf # The pruned workflow is the output
                                self._log_message(f"Successfully processed and saved reference for port '{uid}'.", "debug")
                            else: # pass_by_method == 'val'
                                try:
                                    data = context.load(uid)
                                    final_outputs[uid] = data
                                    context.save(uid, data, use_ram=use_ram, pass_by='val')
                                    self._log_message(f"Successfully processed and saved value for port '{uid}'.", "debug")
                                except KeyError:
                                    self._log_message(f"No output found in context for value port '{uid}'.", "warning")

                    self._log_message(f"Usage report after iteration {iter_num+1}: {context.get_usage()}", "debug")
                return final_outputs
        except Exception as e:
            self._log_message(f"An error occurred during run_sequential: {str(e)}", "error")
            import traceback
            traceback.print_exc()
            raise
        finally:
            self._log_message("run_sequential finished. WorkflowContext will now clean up resources.", "info")