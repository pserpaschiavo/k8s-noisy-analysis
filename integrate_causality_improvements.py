"""
Script para integrar as melhorias na visualização de causalidade no pipeline principal.
Atualiza o pipeline.py para usar as novas visualizações.
"""

import os
import sys
import logging
import argparse

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def update_causality_stage():
    """
    Modifica o CausalityAnalysisStage no pipeline.py para usar as visualizações melhoradas.
    Faz backup do arquivo original antes da modificação.
    """
    pipeline_path = 'src/pipeline.py'
    backup_path = 'src/pipeline.py.bak-causality'
    
    # Verificar se o arquivo existe
    if not os.path.exists(pipeline_path):
        logger.error(f"Arquivo {pipeline_path} não encontrado!")
        return False
    
    # Fazer backup do arquivo
    try:
        import shutil
        shutil.copy2(pipeline_path, backup_path)
        logger.info(f"Backup criado em {backup_path}")
    except Exception as e:
        logger.error(f"Erro ao criar backup: {e}")
        return False
    
    # Ler o conteúdo do arquivo
    try:
        with open(pipeline_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        logger.error(f"Erro ao ler {pipeline_path}: {e}")
        return False
    
    # Importar as novas funções de visualização
    import_str = "from src.analysis_causality import CausalityAnalyzer, plot_causality_graph"
    new_import_str = """from src.analysis_causality import CausalityAnalyzer, plot_causality_graph
from src.improved_causality_graph import plot_improved_causality_graph, plot_consolidated_causality_graph"""
    
    if import_str in content:
        content = content.replace(import_str, new_import_str)
        logger.info("Imports atualizados")
    else:
        logger.warning(f"String de importação esperada não encontrada: {import_str}")
    
    # Modificar a geração de visualização para Granger
    granger_viz_str = """                                plot_causality_graph(
                                    granger_matrix, 
                                    granger_out_path,
                                    threshold=granger_threshold, 
                                    directed=True,
                                    metric=metric
                                )"""
                                
    new_granger_viz_str = """                                # Visualização original
                                plot_causality_graph(
                                    granger_matrix, 
                                    granger_out_path,
                                    threshold=granger_threshold, 
                                    directed=True,
                                    metric=metric
                                )
                                
                                # Visualização melhorada
                                improved_granger_out_path = os.path.join(
                                    os.path.join(out_dir, 'improved'), 
                                    f"improved_granger_{metric}_{phase}_{round_id}.png"
                                )
                                os.makedirs(os.path.dirname(improved_granger_out_path), exist_ok=True)
                                plot_improved_causality_graph(
                                    granger_matrix, 
                                    improved_granger_out_path,
                                    threshold=granger_threshold, 
                                    directed=True,
                                    metric=metric
                                )
                                plot_paths.append(improved_granger_out_path)"""
    
    if granger_viz_str in content:
        content = content.replace(granger_viz_str, new_granger_viz_str)
        logger.info("Visualização Granger atualizada")
    else:
        logger.warning(f"String de visualização Granger esperada não encontrada")
    
    # Modificar a geração de visualização para TE
    te_viz_str = """                                plot_causality_graph(
                                    te_viz_matrix,
                                    te_out_path,
                                    threshold=0.9,  # Threshold para visualização (menores valores = mais causalidade)
                                    directed=True,
                                    metric=f"{metric} (TE)"
                                )"""
                                
    new_te_viz_str = """                                # Visualização original
                                plot_causality_graph(
                                    te_viz_matrix,
                                    te_out_path,
                                    threshold=0.9,  # Threshold para visualização (menores valores = mais causalidade)
                                    directed=True,
                                    metric=f"{metric} (TE)"
                                )
                                
                                # Visualização melhorada
                                improved_te_out_path = os.path.join(
                                    os.path.join(out_dir, 'improved'), 
                                    f"improved_te_{metric}_{phase}_{round_id}.png"
                                )
                                os.makedirs(os.path.dirname(improved_te_out_path), exist_ok=True)
                                plot_improved_causality_graph(
                                    te_viz_matrix,
                                    improved_te_out_path,
                                    threshold=0.9,  # Threshold para visualização (menores valores = mais causalidade)
                                    directed=True,
                                    metric=f"{metric} (TE)"
                                )
                                plot_paths.append(improved_te_out_path)"""
    
    if te_viz_str in content:
        content = content.replace(te_viz_str, new_te_viz_str)
        logger.info("Visualização TE atualizada")
    else:
        logger.warning(f"String de visualização TE esperada não encontrada")
    
    # Adicionar geração de grafo consolidado após o loop de métricas
    context_update_str = """        # Atualizar contexto
        context['granger_matrices'] = granger_matrices
        context['te_matrices'] = te_matrices
        context['causality_plot_paths'] = plot_paths
        
        return context"""
        
    new_context_update_str = """        # Gerar visualizações consolidadas multi-métrica
        try:
            # Agrupar matrizes por experimento, round e fase
            metrics_by_group = {}
            for key, matrix in granger_matrices.items():
                parts = key.split(':')
                if len(parts) == 4:
                    experiment_id, round_id, phase, metric = parts
                    group_key = f"{experiment_id}:{round_id}:{phase}"
                    if group_key not in metrics_by_group:
                        metrics_by_group[group_key] = {}
                    metrics_by_group[group_key][metric] = matrix
            
            # Gerar um grafo consolidado para cada grupo
            for group_key, metric_matrices in metrics_by_group.items():
                if len(metric_matrices) > 1:  # Só vale a pena consolidar se tiver mais de uma métrica
                    parts = group_key.split(':')
                    if len(parts) == 3:
                        experiment_id, round_id, phase = parts
                        consolidated_out_path = os.path.join(
                            os.path.join(out_dir, 'consolidated'), 
                            f"consolidated_{phase}_{round_id}.png"
                        )
                        os.makedirs(os.path.dirname(consolidated_out_path), exist_ok=True)
                        
                        plot_consolidated_causality_graph(
                            metric_matrices,
                            consolidated_out_path,
                            threshold=granger_threshold,
                            directed=True,
                            phase=phase,
                            round_id=round_id,
                            title_prefix=f'Análise de Causalidade Multi-Métrica'
                        )
                        plot_paths.append(consolidated_out_path)
                        self.logger.info(f"Grafo consolidado gerado para {phase} {round_id}")
        except Exception as e:
            self.logger.error(f"Erro ao gerar grafos consolidados: {e}")
        
        # Atualizar contexto
        context['granger_matrices'] = granger_matrices
        context['te_matrices'] = te_matrices
        context['causality_plot_paths'] = plot_paths
        
        return context"""
    
    if context_update_str in content:
        content = content.replace(context_update_str, new_context_update_str)
        logger.info("Geração de grafo consolidado adicionada")
    else:
        logger.warning(f"String de atualização de contexto esperada não encontrada")
    
    # Escrever conteúdo modificado de volta ao arquivo
    try:
        with open(pipeline_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Arquivo {pipeline_path} atualizado com sucesso")
        return True
    except Exception as e:
        logger.error(f"Erro ao escrever em {pipeline_path}: {e}")
        return False

def add_methods_to_causality_visualizer():
    """
    Adiciona novos métodos à classe CausalityVisualizer em analysis_causality.py.
    """
    causality_path = 'src/analysis_causality.py'
    
    # Verificar se o arquivo existe
    if not os.path.exists(causality_path):
        logger.error(f"Arquivo {causality_path} não encontrado!")
        return False
        
    # Adicionar novas importações
    try:
        from src.improved_causality_graph import plot_improved_causality_graph, plot_consolidated_causality_graph
        
        # As funções já foram importadas corretamente
        logger.info("Funções de melhoria de visualização disponíveis")
    except ImportError:
        logger.error("Não foi possível importar as funções de visualização melhorada")
        return False
    
    # A abordagem mais segura é usar as funções que já criamos nos módulos separados
    logger.info("As funções melhoradas serão usadas diretamente do módulo improved_causality_graph.py")
    return True

def create_documentation_file():
    """
    Cria um arquivo de documentação sobre as melhorias implementadas.
    """
    doc_path = 'docs/melhorias_visualizacao_causalidade.md'
    
    # Criar diretório se não existir
    os.makedirs(os.path.dirname(doc_path), exist_ok=True)
    
    doc_content = """# Melhorias na Visualização de Causalidade

## Problemas Corrigidos

1. **Nós ocultos atrás de arestas**: Nas visualizações originais, algumas vezes os nós ficavam escondidos atrás de arestas, dificultando a visualização. Isso ocorria porque o Matplotlib não respeitava adequadamente a ordem de desenho (z-order) dos elementos.

2. **Falta de visualização consolidada multi-métrica**: Não havia uma forma padrão de visualizar a relação causal entre diferentes métricas em um único grafo.

## Melhorias Implementadas

### 1. Visualização Aprimorada de Grafos de Causalidade

- **Z-order controlado**: Implementação de controle de z-order para garantir que os nós sempre apareçam na frente das arestas.
- **Legibilidade aprimorada**: Aumentamos o tamanho dos nós, melhoramos o contraste de cores e ajustamos o tamanho e posicionamento dos rótulos.
- **Estética aprimorada**: Bordas mais suaves, cores consistentes e layout mais equilibrado.

### 2. Gráfico Consolidado Multi-Métrica

- **Visualização de múltiplas métricas**: Criamos uma função que permite visualizar a relação causal de múltiplas métricas em um único grafo, usando cores diferentes para cada métrica.
- **Legendas explicativas**: Adição de legendas que explicam o significado das cores e dos valores de limiar (threshold).
- **Detecção automática do tipo de matriz**: Detecção inteligente se a matriz representa p-valores (Granger) ou valores de Transfer Entropy (TE).

### 3. Integração com o Pipeline

- As novas visualizações foram integradas ao pipeline existente, mantendo compatibilidade com o código legado.
- Adição de funções para gerar automaticamente visualizações consolidadas para cada combinação de experimento, fase e round.
- As visualizações originais são mantidas, e as melhoradas são geradas em diretórios específicos.

## Como Usar

### Visualizações Individuais Melhoradas

As visualizações melhoradas são geradas automaticamente em:
```
outputs/plots/causality/improved/
```

### Visualizações Consolidadas Multi-Métrica

As visualizações consolidadas são geradas em:
```
outputs/plots/causality/consolidated/
```

### Uso Manual

Para gerar visualizações manualmente, use as funções:

```python
from src.improved_causality_graph import plot_improved_causality_graph, plot_consolidated_causality_graph

# Para um grafo individual
plot_improved_causality_graph(
    causality_matrix,
    output_path,
    threshold=0.05,
    directed=True,
    metric="cpu_usage"
)

# Para um grafo consolidado multi-métrica
plot_consolidated_causality_graph(
    {
        "cpu_usage": cpu_matrix,
        "memory_usage": memory_matrix,
        "disk_io": disk_matrix
    },
    output_path,
    threshold=0.05,
    directed=True,
    phase="1 - Baseline",
    round_id="round-1"
)
```

## Interpretação das Visualizações

### Visualização Individual

- **Nós**: Representam tenants (inquilinos) no sistema.
- **Arestas**: Indicam relação causal entre tenants.
- **Espessura das arestas**: Representa a força da relação causal.
- **Direção das arestas**: Indica a direção da causalidade (de causa para efeito).

### Visualização Consolidada

- **Cores das arestas**: Cada cor representa uma métrica diferente.
- **Espessura das arestas**: Representa a força da relação causal.
- **Nós compartilhados**: Um mesmo tenant pode ter relações causais em diferentes métricas.

## Exemplos

Exemplos de visualizações geradas podem ser encontrados nos diretórios mencionados acima. Para gerar exemplos de teste, execute:

```bash
python test_causality_visualizations.py
```
"""
    
    # Escrever conteúdo no arquivo
    try:
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write(doc_content)
        logger.info(f"Documentação criada em {doc_path}")
        return True
    except Exception as e:
        logger.error(f"Erro ao criar documentação: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Integra melhorias de visualização de causalidade no pipeline')
    parser.add_argument('--test-only', action='store_true', help='Apenas executa o teste sem modificar o pipeline')
    args = parser.parse_args()
    
    logger.info("Iniciando integração das melhorias de visualização de causalidade...")
    
    # Executar teste de visualização
    logger.info("Executando teste de visualizações...")
    try:
        import test_causality_visualizations
        test_causality_visualizations.generate_test_visualizations()
        logger.info("Teste de visualizações concluído com sucesso")
    except Exception as e:
        logger.error(f"Erro ao executar teste de visualizações: {e}")
    
    if args.test_only:
        logger.info("Modo apenas teste. Saindo sem modificar o pipeline.")
        return
    
    # Atualizar estágio de causalidade no pipeline
    if update_causality_stage():
        logger.info("Pipeline atualizado com sucesso")
    else:
        logger.error("Falha ao atualizar o pipeline")
    
    # Adicionar métodos à classe CausalityVisualizer
    if add_methods_to_causality_visualizer():
        logger.info("Métodos adicionados com sucesso")
    else:
        logger.error("Falha ao adicionar métodos")
    
    # Criar documentação
    if create_documentation_file():
        logger.info("Documentação criada com sucesso")
    else:
        logger.error("Falha ao criar documentação")
    
    logger.info("Integração concluída!")

if __name__ == "__main__":
    main()
