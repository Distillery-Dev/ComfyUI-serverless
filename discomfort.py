# discomfort.py
from .workflow_tools import WorkflowTools
from .workflow_context import WorkflowContext
from .comfy_serverless import ComfyConnector
from typing import Dict, List, Any, Optional
import copy
import tempfile
import os
import json
import asyncio

class Discomfort:
    """
    Master class that encapsulates all Discomfort functionality into a single,
    developer-friendly interface for programmatically controlling ComfyUI.
    
    This class provides a simplified API by combining WorkflowTools, WorkflowContext,
    and ComfyConnector into a cohesive interface that makes ComfyUI automation
    straightforward and intuitive.
    """
    
    # Expose Context class directly so users can call discomfort.Context()
    Context = WorkflowContext
    
    def __init__(self):
        """
        Lightweight initialization. Creates WorkflowTools instance and sets up
        placeholders for the async components that will be initialized by create().
        """
        self.Tools = WorkflowTools()
        self.Worker = None  # Placeholder for ComfyConnector instance
    
    @classmethod
    async def create(cls, config_path: Optional[str] = None):
        """
        Async factory method to create and initialize a Discomfort instance.
        This is the primary way users should instantiate Discomfort.
        
        Args:
            config_path: Optional path to custom configuration file
            
        Returns:
            Fully initialized Discomfort instance with Worker ready
        """
        instance = cls()
        instance.Worker = await ComfyConnector.create(config_path=config_path)
        return instance
    
    async def shutdown(self):
        """
        Gracefully shuts down the managed ComfyUI worker instance.
        Should be called when done with the Discomfort instance.
        """
        if self.Worker:
            await self.Worker.kill_api()
    
    async def run(self, workflow_paths: List[str], inputs: Optional[Dict[str, Any]] = None, 
                  iterations: int = 1, use_ram: bool = True, 
                  context: Optional[WorkflowContext] = None) -> Dict[str, Any]:
        """
        Execute workflows with state preservation and context management.
        
        This is the main orchestration method that replaces WorkflowTools.run_sequential.
        It handles both single-shot runs and context-managed stateful runs.
        
        Args:
            workflow_paths: List of workflow JSON file paths to execute
            inputs: Optional dictionary mapping unique_ids to input values
            iterations: Number of iterations to run (default: 1)
            use_ram: Whether to prefer RAM storage for context data
            context: Optional external WorkflowContext for stateful runs
            
        Returns:
            Dictionary mapping output unique_ids to their values
        """
        # Hygiene check: if no workflows were given, raise an error
        if not workflow_paths:
            raise ValueError("[Discomfort] (run) No workflows provided")
        
        # Set default inputs if None provided
        if inputs is None:
            inputs = {}
            
        self.Tools._log_message(f'Starting Discomfort.run for {len(workflow_paths)} workflow(s) over {iterations} iteration(s).', 'info')
        
        # Pre-processing: Load workflows and discover ports
        all_ports_info = {}
        all_original_workflows = {}
        for path in workflow_paths:
            with open(path, 'r') as f:
                wf_data = json.load(f)
                all_original_workflows[path] = wf_data
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".json") as temp_f:
                    json.dump(wf_data, temp_f)
                    temp_path = temp_f.name
                try:
                    all_ports_info[path] = self.Tools.discover_port_nodes(temp_path)
                finally:
                    os.remove(temp_path)

        # Create map of all inputs
        all_inputs_map = {}
        for port_info in all_ports_info.values():
            for uid, info in port_info.get('inputs', {}).items():
                if uid not in all_inputs_map:
                    all_inputs_map[uid] = info

        # Context-aware execution logic
        if context is not None:
            # Use external context - for stateful runs across multiple calls
            return await self._run_with_context(
                workflow_paths, inputs, iterations, use_ram, context,
                all_ports_info, all_original_workflows, all_inputs_map
            )
        else:
            # Create temporary context - for simple single-shot runs
            with WorkflowContext() as temp_context:
                return await self._run_with_context(
                    workflow_paths, inputs, iterations, use_ram, temp_context,
                    all_ports_info, all_original_workflows, all_inputs_map
                )
    
    async def _run_with_context(self, workflow_paths: List[str], inputs: Dict[str, Any], 
                               iterations: int, use_ram: bool, context: WorkflowContext,
                               all_ports_info: Dict, all_original_workflows: Dict, 
                               all_inputs_map: Dict) -> Dict[str, Any]:
        """
        Internal method that performs the actual workflow execution with a given context.
        This contains the core logic migrated from WorkflowTools.run_sequential.
        """
        reset_connector = False
        try:
            # Ensure worker is ready
            while self.Worker._state != "ready":
                await asyncio.sleep(0.5)
            self.Tools._log_message(f"Using WorkflowContext with ID: {context.run_id}", "info")
            
            # Add inputs to context
            for unique_id, data in inputs.items():
                port_type = all_inputs_map.get(unique_id, {}).get('type', 'ANY').upper()
                pass_by_method = self.Tools.pass_by_rules.get(port_type, 'val')
                context.save(unique_id, data, use_ram=use_ram, pass_by=pass_by_method)
                self.Tools._log_message(f"Initial input '{unique_id}' saved to context as type '{pass_by_method}'.", "debug")
            
            # Main execution loop
            final_outputs = {}
            
            for iter_num in range(iterations):
                if iterations == 1:
                    self.Tools._log_message(f"--- Starting Sequential Run ---", "info")
                else:
                    self.Tools._log_message(f"--- Starting Sequential Run - Iteration {iter_num + 1}/{iterations} ---", "info")
                
                # Process each workflow
                for path_idx, path in enumerate(workflow_paths):
                    if len(workflow_paths) > 1:
                        self.Tools._log_message(f"Processing workflow {path_idx + 1}/{len(workflow_paths)}: '{os.path.basename(path)}'", "info")
                    else:
                        self.Tools._log_message(f"Processing workflow: '{os.path.basename(path)}'", "info")

                    # Load workflow and port information
                    original_workflow = all_original_workflows[path]
                    port_info = all_ports_info[path]
                    current_workflow = copy.deepcopy(original_workflow)
                    ref_workflows_to_stitch = []
                    uids_handled_by_stitch = set()

                    # Handle pass-by-reference inputs
                    for uid, in_info in port_info['inputs'].items():
                        port_type = in_info.get('type', 'ANY').upper()
                        # check if the port is in the context
                        if context.get_storage_info(uid):
                            pass_by_method = context.get_storage_info(uid).get('pass_by', 'val') # if the port is in the context, use the pass-by method from the context
                            self.Tools._log_message(f"Found input in context: '{uid}'. Using pass-by method: '{pass_by_method}'.", "info")
                        else:
                            pass_by_method = self.Tools.pass_by_rules.get(port_type, 'val')
                            self.Tools._log_message(f"Input '{uid}' not found in context. Using pass-by method: '{pass_by_method}'.", "info")
                        
                        if pass_by_method == 'ref':
                            if context.get_storage_info(uid):
                                self.Tools._log_message(f"Found pass-by-reference input: '{uid}'. Preparing to stitch.", "info")
                                minimal_workflow = context.load(uid)
                                ref_workflows_to_stitch.append(minimal_workflow)
                                uids_handled_by_stitch.add(uid)
                            else:
                                self.Tools._log_message(f"Input '{uid}' is 'ref' type but was not found in context. It will be swapped to a (likely failing) ContextLoader.", "warning")
                    
                    # Stitch reference workflows if needed
                    if ref_workflows_to_stitch:
                        self.Tools._log_message(f"Stitching {len(ref_workflows_to_stitch)} reference workflows...", "info")
                        temp_files = []
                        
                        # Store main workflow
                        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".json") as temp_f:
                            json.dump(current_workflow, temp_f)
                            temp_files.append(temp_f.name)
                        
                        # Store reference workflows
                        for ref_wf in ref_workflows_to_stitch:
                            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".json") as temp_f:
                                json.dump(ref_wf, temp_f)
                                temp_files.append(temp_f.name)
                        
                        try:
                            stitch_result = self.Tools.stitch_workflows(temp_files[1:] + [temp_files[0]])
                            current_workflow = stitch_result['stitched_workflow']
                            # Update port info for stitched workflow
                            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".json") as temp_f:
                                json.dump(current_workflow, temp_f)
                                new_path = temp_f.name
                            port_info = self.Tools.discover_port_nodes(new_path)
                            os.remove(new_path)
                            self.Tools._log_message("Stitching complete. Port info has been updated.", "info")
                        finally:
                            for f in temp_files:
                                os.remove(f)

                    # Convert workflow to prompt and prepare for execution
                    self.Tools._log_message("Converting workflow to prompt JSON...", "debug")
                    prompt = await self.Worker.get_prompt_from_workflow(current_workflow)
                    
                    # Prepare prompt for contextual run
                    modified_prompt = self.Tools._prepare_prompt_for_contextual_run(prompt, port_info, context, uids_handled_by_stitch)

                    # Execute workflow
                    self.Tools._log_message(f"Executing modified prompt for workflow '{os.path.basename(path)}'.", "info")
                    execution_result = await self.Worker.run_workflow(modified_prompt, use_workflow_json=False)
                    
                    if not execution_result:
                        self.Tools._log_message(f"Workflow '{os.path.basename(path)}' execution failed to produce a result. Aborting run.", "error")
                        reset_connector = True
                        raise RuntimeError(f"Workflow '{os.path.basename(path)}' execution failed.")

                    # Process outputs
                    self.Tools._log_message(f"Processing outputs from '{os.path.basename(path)}'...", "debug")
                    for uid, out_info in port_info['outputs'].items():
                        port_type = out_info.get('type', 'ANY').upper()
                        pass_by_method = self.Tools.pass_by_rules.get(port_type, 'val')

                        if pass_by_method == 'ref':
                            self.Tools._log_message(f"Output '{uid}' is pass-by-reference. Pruning workflow.", "info")
                            pruned_wf = self.Tools._prune_workflow_to_output(original_workflow, uid)
                            context.save(uid, pruned_wf, use_ram=use_ram, pass_by='ref')
                            final_outputs[uid] = pruned_wf
                            self.Tools._log_message(f"Successfully processed and saved reference for port '{uid}'.", "debug")
                        else:  # pass_by_method == 'val'
                            try:
                                data = context.load(uid)
                                final_outputs[uid] = data
                                context.save(uid, data, use_ram=use_ram, pass_by='val')
                                self.Tools._log_message(f"Successfully processed and saved value for port '{uid}'.", "debug")
                            except KeyError:
                                self.Tools._log_message(f"No output found in context for value port '{uid}'.", "warning")

                self.Tools._log_message(f"Usage report after iteration {iter_num+1}: {context.get_usage()}", "debug")
            
            return final_outputs
            
        except Exception as e:
            self.Tools._log_message(f"An error occurred during run: {str(e)}", "error")
            import traceback
            traceback.print_exc()
            await self.Worker.kill_api()  # Reset connector on error
            raise
        finally:
            self.Tools._log_message("Discomfort.run finished.", "info")
            if reset_connector:
                await self.Worker.kill_api() 