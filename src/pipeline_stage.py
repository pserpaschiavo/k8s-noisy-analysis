"""
Module: pipeline_stage.py
Description: Defines the base class for all pipeline stages.
"""
import logging
import time
from typing import Dict, Any, Optional
import pandas as pd

class PipelineStage:
    """
    Base class for all stages in the analysis pipeline.
    It defines the common interface and execution logic.
    """
    
    def __init__(self, stage_name: str, description: str):
        """
        Initializes a pipeline stage.
        
        Args:
            stage_name (str): The unique name of the stage.
            description (str): A brief description of what the stage does.
        """
        self.stage_name = stage_name
        self.description = description
        self.logger = logging.getLogger(f"pipeline.{stage_name}")
    
    def execute(self, data: Optional[pd.DataFrame], all_results: Dict[str, Any], round_id: str) -> Dict[str, Any]:
        """
        Executes the main logic of the stage. This method acts as a wrapper.
        
        Args:
            data (Optional[pd.DataFrame]): The primary DataFrame passed from the previous stage.
            all_results (Dict[str, Any]): A dictionary containing all results from previous stages.
            round_id (str): The identifier for the current processing round.
        
        Returns:
            Dict[str, Any]: A dictionary containing the results produced by this stage.
        """
        self.logger.info(f"Starting stage: {self.stage_name} for round: {round_id}")
        start_time = time.time()
        
        try:
            result = self._execute_implementation(data, all_results, round_id)
        except Exception as e:
            self.logger.error(f"Error executing stage {self.stage_name} for round {round_id}: {e}", exc_info=True)
            # Return an empty dictionary or re-raise to halt the pipeline
            return {}
            
        elapsed_time = time.time() - start_time
        self.logger.info(f"Stage {self.stage_name} for round {round_id} completed in {elapsed_time:.2f} seconds")
        
        return result
    
    def _execute_implementation(self, data: Optional[pd.DataFrame], all_results: Dict[str, Any], round_id: str) -> Dict[str, Any]:
        """
        The specific implementation of the stage's logic. This method must be overridden by all subclasses.
        
        Args:
            data (Optional[pd.DataFrame]): The primary DataFrame from the previous stage.
            all_results (Dict[str, Any]): All results from previous stages.
            round_id (str): The current processing round's identifier.
            
        Raises:
            NotImplementedError: If a subclass does not implement this method.
            
        Returns:
            Dict[str, Any]: The results of this stage's execution.
        """
        raise NotImplementedError("Subclasses must implement the _execute_implementation method.")
