
import logging
import time
from typing import Dict, Any

class PipelineStage:
    """Base class for pipeline stages."""
    
    def __init__(self, name: str, description: str):
        """
        Initializes a pipeline stage.
        
        Args:
            name: Name of the stage.
            description: Description of the stage's purpose.
        """
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"pipeline.{name}")
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes this pipeline stage.
        
        Args:
            context: Dictionary with the current pipeline context.
                     Contains data shared between stages.
        
        Returns:
            Updated dictionary with the result of this stage.
        """
        self.logger.info(f"Starting stage: {self.name}")
        start_time = time.time()
        
        result = self._execute_implementation(context)
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Stage {self.name} completed in {elapsed_time:.2f} seconds")
        
        return result
    
    def _execute_implementation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Specific implementation of the stage. Must be overridden by derived classes.
        
        Args:
            context: Current pipeline context.
            
        Returns:
            Updated context after stage execution.
        """
        raise NotImplementedError("Subclasses must implement this method.")
