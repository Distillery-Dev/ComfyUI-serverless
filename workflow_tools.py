import torch
import numpy
from PIL import Image
import json
import networkx as nx
import os
import copy
import tempfile
from typing import Dict, List, Any, Optional
from .workflow_context import WorkflowContext # Import the WorkflowContext for run-specific I/O

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
    
    def open_image_as_tensor(self, image_path) -> torch.Tensor:
        """
        Loads a ComfyUI image from a file.
        Does NOT add the image to the Discomfort context.
        """
        try:
            image = Image.open(image_path)
            image = image.convert("RGB")
            image = numpy.array(image).astype(numpy.float32) / 255.0
            image = torch.from_numpy(image).unsqueeze(0)
            return image
        except Exception as e:
            self._log_message(f"Error opening image as tensor: {e}", "error")
            raise

    def save_comfy_image_to_disk(self, tensor: torch.Tensor, output_path: str):
        """
        Saves a ComfyUI image to a file.
        Does NOT collect the image from the Discomfort context.
        """
        try:
            image = tensor[0]
            image = image * 255
            image = image.clamp(0, 255).to(torch.uint8)
            image_np = image.cpu().numpy()
            pil_image = Image.fromarray(image_np, 'RGB')
            pil_image.save(output_path)
        except Exception as e:
            self._log_message(f"Error saving image to disk: {e}", "error")
            raise
    
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
        """Validate workflow structure for use with Discomfort."""
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

    def _discover_context_handlers(self, workflow: Dict[str, Any]) -> Dict[str, List[Dict]]:
        """
        Discovers internal DiscomfortContextLoader/Saver nodes in a workflow.
        This is used to identify and repair broken nodes from corrupted ref workflows.
        """
        loaders = []
        savers = []
        nodes = workflow.get('nodes', [])

        for node in nodes:
            node_id = node.get('id')
            node_type = node.get('type')
            widgets_values = node.get('widgets_values', [])

            if node_type == "DiscomfortContextLoader" and len(widgets_values) >= 2:
                # Per nodes_internal.py, unique_id is the second widget
                loaders.append({
                    'node_id': node_id,
                    'unique_id': widgets_values[1] 
                })
            elif node_type == "DiscomfortContextSaver" and len(widgets_values) >= 1:
                # Per nodes_internal.py, unique_id is the first widget
                savers.append({
                    'node_id': node_id,
                    'unique_id': widgets_values[0]
                })
                
        return {'loaders': loaders, 'savers': savers}

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

    def stitch_workflows(self, workflows: List[Any], delete_input_ports: bool = False, delete_output_ports: bool = False) -> Dict[str, Any]:
        """
        Stitch multiple workflows together.

        Args:
            workflows (List[Any]): A list of workflows to stitch. Items can be file paths (str)
                                   or workflow objects (dict).
            delete_input_ports (bool): If True, remove all unconnected input ports from the final graph.
            delete_output_ports (bool): If True, remove all unconnected output ports from the final graph.
        """
        import uuid
        import itertools
        
        if not workflows:
            raise ValueError('No workflows provided')

        merged_nodes: list[dict] = []
        merged_links: list[list] = []
        
        next_node_id = itertools.count(1)
        next_link_id = itertools.count(1)

        prev_outputs = {} # Tracks the node_id and type of outputs from previous workflows
        temp_files_to_clean = []

        try:
            for wf_item in workflows:
                temp_path = None
                if isinstance(wf_item, str):
                    path = wf_item
                elif isinstance(wf_item, dict):
                    temp_path = self._create_temp_workflow_from_data(wf_item)
                    temp_files_to_clean.append(temp_path)
                    path = temp_path
                else:
                    raise TypeError(f"Workflow item must be a path (str) or object (dict), not {type(wf_item).__name__}.")

                info = self.discover_port_nodes(path)
                
                old_to_new_id = {old_id: next(next_node_id) for old_id in info['nodes']}

                for old_id, node_data in info['nodes'].items():
                    node = copy.deepcopy(node_data)
                    node['id'] = old_to_new_id[old_id]
                    merged_nodes.append(node)

                old_to_new_link = {}
                for link_data in info.get('links', []):
                    old_link_id = link_data[0]
                    new_link_id = next(next_link_id)
                    old_to_new_link[old_link_id] = new_link_id
                    
                    new_link = link_data.copy()
                    new_link[0] = new_link_id
                    new_link[1] = old_to_new_id.get(link_data[1])
                    new_link[3] = old_to_new_id.get(link_data[3])
                    merged_links.append(new_link)
                
                for node in merged_nodes[-len(info['nodes']):]:
                    if 'inputs' in node:
                        for inp in node.get('inputs', []):
                            if 'link' in inp and inp['link'] is not None:
                                inp['link'] = old_to_new_link.get(inp['link'], inp['link'])
                    if 'outputs' in node:
                        for out in node.get('outputs', []):
                            if 'links' in out and out.get('links') is not None:
                                out['links'] = [old_to_new_link.get(l, l) for l in out['links']]
                
                self._normalize_links(merged_nodes)

                for uid, in_info in info['inputs'].items():
                    if uid in prev_outputs:
                        source_id = prev_outputs[uid]['node_id']
                        target_id = old_to_new_id[in_info['node_id']]
                        link_id = next(next_link_id)
                        
                        new_link = [link_id, source_id, 0, target_id, 0, prev_outputs[uid]['type'] or 'ANY']
                        merged_links.append(new_link)
                        
                        source_node = next((n for n in merged_nodes if n['id'] == source_id), None)
                        if source_node and source_node.get('outputs'):
                            source_node['outputs'][0]['links'].append(link_id)
                        
                        target_node = next((n for n in merged_nodes if n['id'] == target_id), None)
                        if target_node and target_node.get('inputs'):
                            target_node['inputs'][0]['link'] = link_id
                
                for uid, out_info in info['outputs'].items():
                    prev_outputs[uid] = {
                        'node_id': old_to_new_id[out_info['node_id']],
                        'type': out_info['type']
                    }

            # --- Deletion of remaining ports ---
            # Discover the ports of the fully stitched graph before deletion
            temp_stitched_path = self._create_temp_workflow_from_data({'nodes': merged_nodes, 'links': merged_links})
            temp_files_to_clean.append(temp_stitched_path)
            final_info = self.discover_port_nodes(temp_stitched_path)
            
            nodes_to_delete = set()
            if delete_input_ports:
                for port_info in final_info.get('inputs', {}).values():
                    nodes_to_delete.add(port_info['node_id'])
                self._log_message(f"Deleting {len(final_info.get('inputs', {}))} final input ports.", "info")

            if delete_output_ports:
                for port_info in final_info.get('outputs', {}).values():
                    nodes_to_delete.add(port_info['node_id'])
                self._log_message(f"Deleting {len(final_info.get('outputs', {}))} final output ports.", "info")

            if nodes_to_delete:
                merged_nodes = [n for n in merged_nodes if n['id'] not in nodes_to_delete]
                merged_links = [l for l in merged_links if l[1] not in nodes_to_delete and l[3] not in nodes_to_delete]
                
                # After deleting, re-discover the state for the return value
                temp_final_path = self._create_temp_workflow_from_data({'nodes': merged_nodes, 'links': merged_links})
                temp_files_to_clean.append(temp_final_path)
                final_info = self.discover_port_nodes(temp_final_path)
            
            stitched_workflow = {
                'nodes': merged_nodes,
                'links': merged_links,
                'last_node_id': next(next_node_id) - 1,
                'last_link_id': next(next_link_id) - 1,
                'version': 0.4,
            }

            if not self.validate_workflow(stitched_workflow):
                raise ValueError("Stitched workflow validation failed")

            return {
                'stitched_workflow': stitched_workflow,
                'inputs': final_info['inputs'],
                'outputs': final_info['outputs'],
                'execution_order': final_info['execution_order']
            }

        finally:
            # Clean up all temporary files created during the process
            for f_path in temp_files_to_clean:
                try:
                    os.remove(f_path)
                except OSError as e:
                    self._log_message(f"Error removing temporary file {f_path}: {e}", "warning")

    def _delete_passthru_ports(self, workflows: List[Any]) -> List[Dict[str, Any]]:
        """
        Processes a list of workflows to remove all passthrough ports, rewiring
        their connections to maintain workflow integrity. A passthrough port is
        defined as a DiscomfortPort with both an input and an output connection.
    
        This method handles chains of passthrough ports and deduplicates any
        redundant connections that result from the rewiring.
    
        Args:
            workflows: A list of workflows. Each item can be a file path (str)
                       or a workflow object (dict).
    
        Returns:
            A list of modified workflow dictionaries with passthrough ports removed.
        """
        processed_workflows = []
        temp_files_to_clean = []
        try:
            for i, wf_item in enumerate(workflows):
                workflow = None
                if isinstance(wf_item, str):
                    try:
                        with open(wf_item, 'r') as f:
                            workflow = json.load(f)
                    except Exception as e:
                        self._log_message(f"Failed to load workflow from path '{wf_item}': {e}", "error")
                        continue
                elif isinstance(wf_item, dict):
                    workflow = copy.deepcopy(wf_item)
                else:
                    self._log_message(f"Workflow item at index {i} has invalid type: {type(wf_item).__name__}. Skipping.", "warning")
                    continue
    
                # --- Core Deletion Logic ---
    
                # We need discover_port_nodes, which requires a file path.
                temp_path = self._create_temp_workflow_from_data(workflow)
                temp_files_to_clean.append(temp_path)
    
                port_info = self.discover_port_nodes(temp_path)
                passthru_node_ids = {p['node_id'] for p in port_info.get('passthrus', {}).values()}
    
                if not passthru_node_ids:
                    self._log_message("No passthrough ports found to delete in workflow.", "debug")
                    processed_workflows.append(workflow)
                    continue
    
                self._log_message(f"Found {len(passthru_node_ids)} passthrough ports to remove.", "info")
    
                links_list = workflow['links']
                nodes_dict = {node['id']: node for node in workflow['nodes']}
    
                memo = {}
                def find_true_source(node_id, slot_id):
                    if (node_id, slot_id) in memo: return memo[(node_id, slot_id)]
                    if node_id in passthru_node_ids:
                        incoming_link = next((l for l in links_list if l[3] == node_id and l[4] == 0), None)
                        if incoming_link:
                            result = find_true_source(incoming_link[1], incoming_link[2])
                            memo[(node_id, slot_id)] = result
                            return result
                    result = (node_id, slot_id)
                    memo[(node_id, slot_id)] = result
                    return result
    
                rewired_links = []
                for link in links_list:
                    if link[3] not in passthru_node_ids:
                        true_source_id, true_source_slot = find_true_source(link[1], link[2])
                        true_source_node = nodes_dict.get(true_source_id)
                        new_link_type = 'ANY'
                        if true_source_node and 'outputs' in true_source_node and len(true_source_node['outputs']) > true_source_slot:
                            new_link_type = true_source_node['outputs'][true_source_slot].get('type', 'ANY') or 'ANY'
                        rewired_links.append([link[0], true_source_id, true_source_slot, link[3], link[4], new_link_type])
    
                final_nodes = [node for node in workflow['nodes'] if node['id'] not in passthru_node_ids]
    
                unique_connections = {}
                for link in rewired_links:
                    conn_key = (link[1], link[2], link[3], link[4])
                    unique_connections[conn_key] = link[5]
    
                final_links = []
                link_id_counter = max([l[0] for l in workflow.get('links', []) if l] + [0]) + 1
                for (src_id, src_slot, tgt_id, tgt_slot), link_type in unique_connections.items():
                    final_links.append([link_id_counter, src_id, src_slot, tgt_id, tgt_slot, link_type])
                    link_id_counter += 1
    
                links_by_target = {(link[3], link[4]): link[0] for link in final_links}
                for node in final_nodes:
                    if 'inputs' in node:
                        for i_slot, node_input in enumerate(node.get('inputs', [])):
                            slot_index = node_input.get('slot_index', i_slot)
                            node_input['link'] = links_by_target.get((node['id'], slot_index))
    
                new_workflow = {
                    **{k: v for k, v in workflow.items() if k not in ['nodes', 'links']},
                    'nodes': final_nodes,
                    'links': final_links
                }
                processed_workflows.append(new_workflow)
    
        finally:
            for f_path in temp_files_to_clean:
                try:
                    os.remove(f_path)
                except OSError as e:
                    self._log_message(f"Error removing temp file {f_path}: {e}", "warning")
    
        return processed_workflows

    def _normalize_links(self, nodes: list[dict]) -> None:
        """
        Guarantee every node.outputs[*].links is a mutable list (never None).
        This method operates in-place.
        """
        for node in nodes:
            for out in node.get("outputs", []):
                if "links" not in out or out["links"] is None:
                    out["links"] = []

    def _create_temp_workflow_from_data(self, wf_data: dict) -> str:
        """Helper to write workflow data to a temporary file and return the path."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".json", encoding='utf-8') as temp_f:
            json.dump(wf_data, temp_f)
            # Ensure file is closed before returning path on all OSes
            temp_path = temp_f.name
        # The file will be cleaned up by the caller or when the program exits
        return temp_path

    def _prepare_prompt_for_contextual_run(self, prompt: Dict[str, Any], port_info: Dict[str, Any], context: WorkflowContext, pass_by_behaviors: Dict[str, str], handlers_info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Modifies a prompt to work with WorkflowContext by performing class-swapping.
        This version uses both port_info and handlers_info to robustly repair the prompt.
        """
        self._log_message("Preparing prompt for contextual run via class-swapping and repair...", "debug")
        modified_prompt = copy.deepcopy(prompt)
        run_id = context.run_id

        # --- Step 1: Create unified maps of all ports and handlers that need swapping ---
        nodes_to_swap = {}

        # Add swappable DiscomfortPorts
        for unique_id, p_info in {**port_info.get('inputs', {}), **port_info.get('outputs', {})}.items():
            if pass_by_behaviors.get(unique_id, 'val') == 'val':
                nodes_to_swap[str(p_info['node_id'])] = unique_id

        # Add swappable (broken) handlers from stitched workflows
        if handlers_info:
            for handler_type in ['loaders', 'savers']:
                for handler in handlers_info.get(handler_type, []):
                    if pass_by_behaviors.get(handler['unique_id'], 'val') == 'val':
                        nodes_to_swap[str(handler['node_id'])] = handler['unique_id']

        # --- Step 2: Iterate through the prompt and swap/repair nodes ---
        for node_id_str, unique_id in nodes_to_swap.items():
            # Determine if the port is for input or output using the original port_info
            is_input = any(str(p['node_id']) == node_id_str for p in port_info.get('inputs', {}).values())
            is_output = any(str(p['node_id']) == node_id_str for p in port_info.get('outputs', {}).values())
            # A broken handler might not be in port_info, so we check its class type
            is_handler_loader = prompt.get(node_id_str, {}).get('class_type') == 'DiscomfortContextLoader'

            if is_input or is_handler_loader:
                self._log_message(f"Swapping/repairing INPUT port '{unique_id}' (node {node_id_str}) with DiscomfortContextLoader.", "debug")
                modified_prompt[node_id_str] = {
                    "inputs": {"run_id": run_id, "unique_id": unique_id},
                    "class_type": "DiscomfortContextLoader"
                }
            elif is_output: # Savers must be in port_info to get the data link
                self._log_message(f"Swapping/repairing OUTPUT port '{unique_id}' (node {node_id_str}) with DiscomfortContextSaver.", "debug")
                data_input = prompt.get(node_id_str, {}).get('inputs', {}).get('input_data')
                modified_prompt[node_id_str] = {
                    "inputs": {"input_data": data_input, "unique_id": unique_id, "run_id": run_id},
                    "class_type": "DiscomfortContextSaver"
                }
                
        return modified_prompt