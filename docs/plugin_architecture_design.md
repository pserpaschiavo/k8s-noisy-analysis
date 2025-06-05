# Design da Arquitetura Baseada em Plugins para o Pipeline de Análise

## Visão Geral

Este documento detalha o design técnico para a migração do pipeline atual para uma arquitetura baseada em plugins, permitindo maior flexibilidade, extensibilidade e manutenção.

## Motivação

Atualmente, o pipeline possui múltiplas implementações (`pipeline.py`, `pipeline_new.py`, `pipeline_with_sliding_window.py`) que compartilham funcionalidade comum mas foram desenvolvidas de forma independente. Isso causa:

- Duplicação de código
- Inconsistência na implementação
- Dificuldade de manutenção
- Complexidade para adicionar novos tipos de análise

A arquitetura baseada em plugins visa resolver esses problemas ao:

1. Unificar o núcleo do pipeline
2. Modularizar os componentes de análise
3. Permitir carregamento dinâmico de estágios
4. Simplificar a adição de novos tipos de análise

## Design da Arquitetura

### 1. Core do Pipeline

O núcleo do pipeline será responsável por:

- Carregamento de configuração
- Registro de plugins
- Orquestração da execução
- Gerenciamento de contexto
- Verificação de dependências
- Log centralizado

```python
class PipelineCore:
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        self.plugins = {}
        self.context = {}
        self.logger = self._setup_logging()
    
    def register_plugin(self, plugin_class):
        """Registra um plugin no pipeline."""
        plugin_id = plugin_class.get_id()
        if plugin_id in self.plugins:
            self.logger.warning(f"Plugin {plugin_id} já registrado, substituindo.")
        self.plugins[plugin_id] = plugin_class(self.config, self.logger)
    
    def run(self, selected_plugins=None):
        """Executa o pipeline com os plugins selecionados."""
        plugins_to_run = self._resolve_dependencies(selected_plugins)
        for plugin_id in plugins_to_run:
            plugin = self.plugins[plugin_id]
            self.logger.info(f"Executando plugin: {plugin_id}")
            self.context = plugin.execute(self.context)
        return self.context
```

### 2. Interface de Plugin

Todos os plugins devem implementar uma interface comum:

```python
class PipelinePlugin:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        
    @classmethod
    def get_id(cls):
        """Retorna o identificador único do plugin."""
        raise NotImplementedError
    
    @classmethod
    def get_dependencies(cls):
        """Retorna a lista de dependências (IDs de outros plugins)."""
        return []
    
    def execute(self, context):
        """Executa a lógica do plugin e retorna o contexto atualizado."""
        raise NotImplementedError
```

### 3. Sistema de Descoberta de Plugins

Para permitir o carregamento dinâmico de plugins:

```python
def discover_plugins(plugins_dir):
    """Descobre plugins disponíveis no diretório de plugins."""
    plugins = {}
    for module_file in os.listdir(plugins_dir):
        if module_file.endswith('.py') and not module_file.startswith('_'):
            module_name = module_file[:-3]
            module = importlib.import_module(f"plugins.{module_name}")
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, PipelinePlugin) and obj != PipelinePlugin:
                    plugins[obj.get_id()] = obj
    return plugins
```

### 4. Conversão dos Estágios Atuais para Plugins

Cada estágio atual do pipeline será convertido em um plugin:

- `DataIngestionPlugin`: Responsável pela ingestão de dados
- `DescriptiveAnalysisPlugin`: Análise descritiva
- `CorrelationAnalysisPlugin`: Análise de correlação
- `CausalityAnalysisPlugin`: Análise de causalidade
- `SlidingWindowAnalysisPlugin`: Análise com janelas deslizantes
- `MultiRoundAnalysisPlugin`: Análise multi-round
- `InsightAggregationPlugin`: Agregação de insights
- `ReportGenerationPlugin`: Geração de relatórios

## Plano de Implementação

### Fase 1: Estrutura Base (Semana 1)

1. Implementar `PipelineCore`
2. Definir interface `PipelinePlugin`
3. Implementar sistema de descoberta
4. Criar estrutura de diretório `plugins/`
5. Implementar plugin de teste

### Fase 2: Migração (Semana 2)

1. Converter `DataIngestionStage` para `DataIngestionPlugin`
2. Converter `AnalysisDescriptiveStage` para `DescriptiveAnalysisPlugin` 
3. Converter `AnalysisCorrelationStage` para `CorrelationAnalysisPlugin`
4. Converter `AnalysisCausalityStage` para `CausalityAnalysisPlugin`
5. Implementar script principal para executar o pipeline baseado em plugins

### Fase 3: Recursos Avançados (Semana 3-4)

1. Implementar sistema de dependências
2. Integrar paralelização para plugins independentes
3. Adicionar validação de configuração
4. Implementar sistema de cache para resultados
5. Converter plugins restantes (sliding window, multi-round)

## Exemplo de Uso

```python
# Carrega o pipeline com a configuração
pipeline = PipelineCore("config/pipeline_config.yaml")

# Descobre plugins disponíveis
discovered_plugins = discover_plugins("plugins/")
for plugin_id, plugin_class in discovered_plugins.items():
    pipeline.register_plugin(plugin_class)

# Executa o pipeline completo
result = pipeline.run()

# Ou executa apenas análise descritiva e correlação
result = pipeline.run(["descriptive_analysis", "correlation_analysis"])
```

## Considerações para Compatibilidade Retroativa

Para garantir uma transição suave, o sistema manterá compatibilidade com os scripts existentes:

1. Implementação de Wrappers: Criar wrappers sobre a nova arquitetura para manter os scripts existentes funcionais
2. Mapeamento de Configuração: Conversor automático dos formatos antigos de configuração para o novo formato
3. Adaptadores de Output: Garantir que os outputs gerados pelos plugins sejam compatíveis com o esperado pelos scripts existentes

## Próximos Passos

1. Revisão do design por stakeholders
2. Setup do ambiente de desenvolvimento e testes
3. Implementação da estrutura base
4. Migração dos estágios existentes
5. Testes extensivos com os mesmos dados de entrada para comparar resultados
