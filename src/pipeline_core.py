"""
Pipeline core module for the plugin-based architecture.
This module implements the core functionality for the plugin-based pipeline system.
"""
import os
import sys
import logging
import importlib
import inspect
import yaml
from typing import Dict, List, Any, Optional, Type

# Base Plugin Interface
class PipelinePlugin:
    """Base class for all pipeline plugins."""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize the plugin.
        
        Args:
            config: The pipeline configuration
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
    
    @classmethod
    def get_id(cls) -> str:
        """
        Returns the unique identifier for this plugin.
        
        Returns:
            str: The plugin identifier
        """
        raise NotImplementedError("Plugin must implement get_id()")
    
    @classmethod
    def get_dependencies(cls) -> List[str]:
        """
        Returns the list of plugin dependencies (plugin IDs).
        
        Returns:
            List[str]: List of plugin IDs that this plugin depends on
        """
        return []
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the plugin's logic.
        
        Args:
            context: The pipeline context containing data from previous stages
            
        Returns:
            Dict[str, Any]: The updated pipeline context
        """
        raise NotImplementedError("Plugin must implement execute()")

# Pipeline Core
class PipelineCore:
    """Core class for managing the plugin-based pipeline."""
    
    def __init__(self, config_path: str):
        """
        Initialize the pipeline core.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.plugins: Dict[str, PipelinePlugin] = {}
        self.context: Dict[str, Any] = {}
        self.logger = self._setup_logging()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Dict[str, Any]: The configuration dictionary
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _setup_logging(self) -> logging.Logger:
        """
        Set up logging for the pipeline.
        
        Returns:
            logging.Logger: Configured logger
        """
        logger = logging.getLogger('pipeline')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Console handler
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            logger.addHandler(ch)
            
            # File handler
            log_dir = self.config.get('log_dir', 'logs')
            os.makedirs(log_dir, exist_ok=True)
            fh = logging.FileHandler(os.path.join(log_dir, 'pipeline.log'))
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        
        return logger
    
    def register_plugin(self, plugin_class: Type[PipelinePlugin]) -> None:
        """
        Register a plugin with the pipeline.
        
        Args:
            plugin_class: The plugin class to register
        """
        plugin_id = plugin_class.get_id()
        if plugin_id in self.plugins:
            self.logger.warning(f"Plugin {plugin_id} already registered, replacing")
        
        self.plugins[plugin_id] = plugin_class(self.config, self.logger)
        self.logger.info(f"Registered plugin: {plugin_id}")
    
    def _resolve_dependencies(self, selected_plugins: Optional[List[str]] = None) -> List[str]:
        """
        Resolve plugin dependencies and return an ordered list of plugins to execute.
        
        Args:
            selected_plugins: List of selected plugin IDs to run. If None, all plugins are run.
            
        Returns:
            List[str]: Ordered list of plugin IDs to execute
        """
        if selected_plugins is None:
            selected_plugins = list(self.plugins.keys())
        
        # Check that all selected plugins are registered
        for plugin_id in selected_plugins:
            if plugin_id not in self.plugins:
                raise ValueError(f"Plugin {plugin_id} is not registered")
        
        # Build dependency graph
        dependency_graph = {}
        for plugin_id in selected_plugins:
            plugin = self.plugins[plugin_id]
            deps = [dep for dep in plugin.get_dependencies() if dep in selected_plugins]
            dependency_graph[plugin_id] = deps
        
        # Topological sort
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(plugin_id):
            if plugin_id in temp_visited:
                raise ValueError(f"Circular dependency detected involving {plugin_id}")
            
            if plugin_id not in visited:
                temp_visited.add(plugin_id)
                for dep in dependency_graph[plugin_id]:
                    visit(dep)
                temp_visited.remove(plugin_id)
                visited.add(plugin_id)
                order.append(plugin_id)
        
        # Visit all nodes
        for plugin_id in dependency_graph:
            if plugin_id not in visited:
                visit(plugin_id)
        
        return order
    
    def run(self, selected_plugins: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run the pipeline with the selected plugins.
        
        Args:
            selected_plugins: List of plugin IDs to run. If None, all plugins are run.
            
        Returns:
            Dict[str, Any]: The final pipeline context
        """
        self.logger.info("Starting pipeline execution")
        
        try:
            plugins_to_run = self._resolve_dependencies(selected_plugins)
            self.logger.info(f"Resolved execution order: {plugins_to_run}")
            
            for plugin_id in plugins_to_run:
                plugin = self.plugins[plugin_id]
                self.logger.info(f"Executing plugin: {plugin_id}")
                try:
                    self.context = plugin.execute(self.context)
                except Exception as e:
                    self.logger.error(f"Error executing plugin {plugin_id}: {e}")
                    raise
                
            self.logger.info("Pipeline execution completed successfully")
            return self.context
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            raise
            
def discover_plugins(plugins_dir: str) -> Dict[str, Type[PipelinePlugin]]:
    """
    Discover plugins available in the plugins directory.
    
    Args:
        plugins_dir: Path to the plugins directory
        
    Returns:
        Dict[str, Type[PipelinePlugin]]: Dictionary mapping plugin IDs to plugin classes
    """
    plugins = {}
    sys.path.append(os.path.dirname(plugins_dir))
    plugins_package = os.path.basename(plugins_dir)
    
    for module_file in os.listdir(plugins_dir):
        if module_file.endswith('.py') and not module_file.startswith('_'):
            module_name = module_file[:-3]
            try:
                module = importlib.import_module(f"{plugins_package}.{module_name}")
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, PipelinePlugin) and 
                        obj != PipelinePlugin):
                        try:
                            plugin_id = obj.get_id()
                            plugins[plugin_id] = obj
                        except NotImplementedError:
                            # Skip abstract classes that don't implement get_id
                            continue
            except ImportError as e:
                logging.warning(f"Failed to import plugin module {module_name}: {e}")
    
    return plugins
