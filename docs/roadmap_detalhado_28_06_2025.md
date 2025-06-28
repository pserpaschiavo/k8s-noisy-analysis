# Roadmap Atualizado - Pipeline K8s-Noisy-Analysis
**Data: 28 de Junho de 2025**

## 📊 **Estado Atual do Pipeline**

Após análise detalhada do código, foi identificado que:

- ✅ **Pipeline base**: Funcional com 10 etapas de processamento
- ✅ **Análise Multi-Round**: Framework implementado, mas com problemas nas visualizações
- ✅ **Time series consolidados**: Implementados com sucesso em `advanced_plots.py`
- ❌ **Visualizações específicas**: 3 problemas críticos identificados

### **Problemas Identificados**

1. **Grafos de Correlação**: Função `plot_aggregated_correlation_graph` mencionada nos planos, mas não implementada no código
2. **Séries Temporais**: Algumas visualizações não estão sendo geradas corretamente
3. **Heatmaps de Causalidade**: Geração inconsistente, funcionando apenas intermitentemente

---

## 🔥 **FASE 1: CORREÇÃO DE BUGS DE VISUALIZAÇÃO** 
**Meta para hoje (28/06/2025)** | **Prioridade**: MÁXIMA

### **1.1 Corrigir Grafos de Correlação**
- [ ] **Implementar a função `plot_aggregated_correlation_graph`** em `src/visualization/advanced_plots.py`
  - [ ] Implementar visualização de grafo baseada em NetworkX
  - [ ] Configurar pesos de arestas baseados em valores de correlação
  - [ ] Criar esquema de cores e legendas para facilitar interpretação
  - [ ] Adicionar parâmetro de threshold para filtrar correlações fracas

### **1.2 Restaurar Geração de Séries Temporais**
- [ ] **Adicionar verificações de dados** nas funções de séries temporais
  - [ ] Verificar DataFrame vazio ou com dados insuficientes
  - [ ] Normalizar manipulação de timestamps entre diferentes funções
  - [ ] Adicionar logging detalhado para diagnóstico de falhas

### **1.3 Resolver Inconsistências de Heatmaps**
- [ ] **Reforçar validação de dados** na função `plot_causality_heatmap`
  - [ ] Validar formato e dimensões das matrizes de causalidade
  - [ ] Adicionar tratamento para valores extremos ou ausentes
  - [ ] Implementar escala adaptativa para melhor visualização

---

## 🛡️ **FASE 2: VALIDAÇÃO DE DADOS (Hoje à tarde)**
**Duração**: 3-4 horas | **Prioridade**: ALTA

### **2.1 Implementar Framework de Validação**
- [ ] **Criar módulo `src/validation.py`**
  - [ ] Implementar função `validate_data_for_visualization`
  - [ ] Criar validadores específicos para cada tipo de visualização
  - [ ] Definir respostas padrão para dados inválidos (gráficos vazios vs. erro)

### **2.2 Integrar Validação em Todos os Módulos**
- [ ] **Adicionar chamadas de validação** em todas as funções de plotagem
  - [ ] Integrar em funções de `plots.py`
  - [ ] Integrar em funções de `advanced_plots.py`
  - [ ] Garantir feedback claro ao usuário em caso de falha

---

## 🏗️ **FASE 3: VISUALIZAÇÕES CONSOLIDADAS (Final do dia)**
**Duração**: 2-3 horas | **Prioridade**: MÉDIA

### **3.1 Unificar Visualizações de Comparação de Fase**
- [ ] **Implementar visualização consolidada** para comparação entre rounds
  - [ ] Criar layouts com múltiplos subplots (2x2)
  - [ ] Organizar visualizações por métrica
  - [ ] Adicionar formatação publication-ready

### **3.2 Teste Completo do Pipeline**
- [ ] **Executar pipeline completo** com todas as correções
  - [ ] Verificar geração de todos os gráficos esperados
  - [ ] Validar qualidade e legibilidade das visualizações
  - [ ] Registrar e corrigir quaisquer inconsistências remanescentes

---

## 📌 **Próximos Passos Imediatos**

1. **Iniciar pela implementação** da função `plot_aggregated_correlation_graph`
2. **Adicionar validações de dados** nas funções problemáticas
3. **Testar cada correção** isoladamente antes de prosseguir
4. **Documentar todas as alterações** com comentários claros no código

---

## 🔄 **Cronograma para Hoje (28/06)**

| Horário | Atividade |
|---------|-----------|
| 09:00-11:00 | Implementar função de grafos de correlação |
| 11:00-12:30 | Corrigir geração de séries temporais |
| 14:00-15:30 | Resolver inconsistências de heatmaps |
| 15:30-17:00 | Implementar framework de validação |
| 17:00-18:30 | Testes finais e documentação |

Este plano será revisado ao final do dia para registrar o progresso e ajustar as próximas etapas conforme necessário.
