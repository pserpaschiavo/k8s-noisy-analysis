#!/usr/bin/env python3
"""
Module: pipeline_plugin.py
Description: Framework de plugins para o pipeline de análise de séries temporais.

Este módulo implementa uma infraestrutura baseada em plugins para o pipeline,
permitindo a adição de novos estágios de análise de forma modular e desacoplada.
É a base para a futura arquitetura unificada do pipeline.
"""

import os
import sys
import importlib
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Type, Union

# Configuração de logging
logger = logging.getLogger(__name__)

class PipelinePlugin(ABC):
    """Classe base para todos os plugins do pipeline."""
    
    def __init__(self, name: str, description: str):
        """
        Inicializa um plugin do pipeline.
        
        Args:
            name: Nome do plugin.
            description: Descrição da funcionalidade do plugin.
        """
        self.name = name
        self.description = description
        self.enabled = True
        self.priority = 100  # Prioridade padrão, valores menores executam primeiro
    
    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executa a funcionalidade do plugin.
        
        Args:
            context: Contexto atual do pipeline.
            
        Returns:
            Contexto atualizado após a execução do plugin.
        """
        pass
    
    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configura o plugin com base em um dicionário de configuração.
        
        Args:
            config: Dicionário com configuração específica para o plugin.
        """
        if 'enabled' in config:
            self.enabled = bool(config['enabled'])
        
        if 'priority' in config:
            self.priority = int(config['priority'])
    
    def __str__(self) -> str:
        status = "Enabled" if self.enabled else "Disabled"
        return f"{self.name} ({status}, priority={self.priority}): {self.description}"

class VisualizationPlugin(PipelinePlugin):
    """Plugin específico para geração de visualizações."""
    
    def __init__(self, name: str, description: str, output_dir: Optional[str] = None):
        """
        Inicializa um plugin de visualização.
        
        Args:
            name: Nome do plugin.
            description: Descrição da funcionalidade do plugin.
            output_dir: Diretório para salvar as visualizações.
        """
        super().__init__(name, description)
        self.output_dir = output_dir or "outputs/plots"
        self.visualization_types = []
    
    def create_output_directory(self) -> None:
        """Cria o diretório de saída para as visualizações, se não existir."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Criado diretório de saída: {self.output_dir}")
    
    def register_visualization_type(self, viz_type: str) -> None:
        """
        Registra um tipo de visualização produzido por este plugin.
        
        Args:
            viz_type: Tipo/nome da visualização.
        """
        if viz_type not in self.visualization_types:
            self.visualization_types.append(viz_type)
    
    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configura o plugin com base em um dicionário de configuração.
        
        Args:
            config: Dicionário com configuração específica para o plugin.
        """
        super().configure(config)
        
        if 'output_dir' in config:
            self.output_dir = config['output_dir']
            
            # Normaliza path do diretório
            self.output_dir = os.path.normpath(self.output_dir)
            
            # Cria diretório se não existir
            self.create_output_directory()

class AnalysisPlugin(PipelinePlugin):
    """Plugin para análises que não geram visualizações diretamente."""
    
    def __init__(self, name: str, description: str):
        super().__init__(name, description)
        self.analysis_types = []
    
    def register_analysis_type(self, analysis_type: str) -> None:
        """
        Registra um tipo de análise produzido por este plugin.
        
        Args:
            analysis_type: Tipo/nome da análise.
        """
        if analysis_type not in self.analysis_types:
            self.analysis_types.append(analysis_type)

class PluginRegistry:
    """Registro central de plugins disponíveis para o pipeline."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PluginRegistry, cls).__new__(cls)
            cls._instance._plugins = {}
            cls._instance._plugin_types = {}
        return cls._instance
    
    def register_plugin(self, plugin_id: str, plugin: PipelinePlugin) -> None:
        """
        Registra um plugin no registro.
        
        Args:
            plugin_id: Identificador único do plugin.
            plugin: Instância do plugin.
        """
        self._plugins[plugin_id] = plugin
        
        # Registra o tipo do plugin
        plugin_type = type(plugin).__name__
        if plugin_type not in self._plugin_types:
            self._plugin_types[plugin_type] = []
        
        if plugin_id not in self._plugin_types[plugin_type]:
            self._plugin_types[plugin_type].append(plugin_id)
        
        logger.info(f"Plugin registrado: {plugin_id} ({plugin_type})")
    
    def get_plugin(self, plugin_id: str) -> Optional[PipelinePlugin]:
        """
        Obtém um plugin pelo seu identificador.
        
        Args:
            plugin_id: Identificador único do plugin.
            
        Returns:
            Instância do plugin ou None se não encontrado.
        """
        return self._plugins.get(plugin_id)
    
    def get_plugins_by_type(self, plugin_type: str) -> List[PipelinePlugin]:
        """
        Obtém todos os plugins de um determinado tipo.
        
        Args:
            plugin_type: Nome do tipo de plugin.
            
        Returns:
            Lista de instâncias de plugins do tipo especificado.
        """
        plugin_ids = self._plugin_types.get(plugin_type, [])
        return [self._plugins[plugin_id] for plugin_id in plugin_ids]
    
    def get_enabled_plugins(self) -> List[PipelinePlugin]:
        """
        Obtém todos os plugins habilitados, ordenados por prioridade.
        
        Returns:
            Lista de instâncias de plugins habilitados.
        """
        enabled_plugins = [p for p in self._plugins.values() if p.enabled]
        return sorted(enabled_plugins, key=lambda p: p.priority)
    
    def load_plugins_from_module(self, module_path: str) -> None:
        """
        Carrega plugins a partir de um módulo Python.
        
        Args:
            module_path: Caminho para o módulo (dotted path).
        """
        try:
            module = importlib.import_module(module_path)
            
            # Procura por subclasses de PipelinePlugin
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                
                if isinstance(attr, type) and issubclass(attr, PipelinePlugin) and attr != PipelinePlugin:
                    # Verifica se a classe tem uma função de registro
                    if hasattr(attr, 'register_plugin'):
                        attr.register_plugin()
            
            logger.info(f"Plugins carregados do módulo: {module_path}")
        except ImportError as e:
            logger.error(f"Erro ao importar módulo de plugins {module_path}: {e}")

class UnifiedPipeline:
    """
    Implementação unificada do pipeline baseada em plugins.
    Esta classe será o novo ponto central do sistema.
    """
    
    def __init__(self):
        """Inicializa um novo pipeline unificado."""
        self.registry = PluginRegistry()
        self.context = {"config": {}, "results": {}, "metadata": {}}
    
    def load_plugins(self, plugin_modules: List[str]) -> None:
        """
        Carrega plugins de módulos especificados.
        
        Args:
            plugin_modules: Lista de caminhos para módulos com plugins.
        """
        for module_path in plugin_modules:
            self.registry.load_plugins_from_module(module_path)
    
    def configure_plugins(self, config: Dict[str, Any]) -> None:
        """
        Configura todos os plugins registrados.
        
        Args:
            config: Dicionário de configuração para todos os plugins.
        """
        plugin_configs = config.get('plugins', {})
        
        for plugin_id, plugin_config in plugin_configs.items():
            plugin = self.registry.get_plugin(plugin_id)
            if plugin:
                plugin.configure(plugin_config)
                logger.info(f"Plugin configurado: {plugin_id}")
            else:
                logger.warning(f"Plugin não encontrado para configuração: {plugin_id}")
    
    def run(self) -> Dict[str, Any]:
        """
        Executa o pipeline com todos os plugins habilitados.
        
        Returns:
            Contexto final após a execução de todos os plugins.
        """
        logger.info("Iniciando execução do pipeline unificado")
        
        # Obtém plugins habilitados ordenados por prioridade
        plugins = self.registry.get_enabled_plugins()
        logger.info(f"Executando {len(plugins)} plugins")
        
        # Executa cada plugin em ordem
        for plugin in plugins:
            logger.info(f"Executando plugin: {plugin.name}")
            try:
                self.context = plugin.execute(self.context)
                logger.info(f"Plugin concluído: {plugin.name}")
            except Exception as e:
                logger.error(f"Erro ao executar plugin {plugin.name}: {e}", exc_info=True)
        
        logger.info("Execução do pipeline concluída")
        return self.context

# Exemplos de como os plugins seriam implementados para o novo sistema
class DescriptiveAnalysisPlugin(VisualizationPlugin):
    """Plugin para análise descritiva."""
    
    def __init__(self, output_dir: Optional[str] = None):
        super().__init__(
            name="Descriptive Analysis",
            description="Análise descritiva e visualização de séries temporais",
            output_dir=output_dir or "outputs/plots/descriptive"
        )
        self.register_visualization_type("timeseries_multi")
        self.register_visualization_type("barplot")
        self.register_visualization_type("boxplot")
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # Implementação seria migrada do DescriptiveAnalysisStage
        logger.info("Executando análise descritiva")
        return context

class CorrelationAnalysisPlugin(VisualizationPlugin):
    """Plugin para análise de correlação."""
    
    def __init__(self, output_dir: Optional[str] = None):
        super().__init__(
            name="Correlation Analysis",
            description="Análise de correlação entre métricas de diferentes tenants",
            output_dir=output_dir or "outputs/plots/correlation"
        )
        self.register_visualization_type("correlation_heatmap")
        self.register_visualization_type("covariance_heatmap")
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # Implementação seria migrada do CorrelationAnalysisStage
        logger.info("Executando análise de correlação")
        return context

# Registro dos plugins
def register_core_plugins(registry: PluginRegistry) -> None:
    """
    Registra os plugins principais no registro.
    
    Args:
        registry: Instância do registro de plugins.
    """
    registry.register_plugin("descriptive", DescriptiveAnalysisPlugin())
    registry.register_plugin("correlation", CorrelationAnalysisPlugin())
    # Mais plugins seriam registrados aqui...
