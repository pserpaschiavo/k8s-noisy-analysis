#!/usr/bin/env python3
"""
Simple script to fix TypeError issues in insight_aggregation.py
"""
import re

with open('src/insight_aggregation.py', 'r') as file:
    content = file.read()

# Fix bool() calls on DataFrame-like objects
content = re.sub(r'(phase_comparison_results|granger_matrices|te_matrices|correlation_matrices|anomaly_metrics) is not None and bool\(\1\)', r'\1 is not None', content)

# Fix dictionary check pattern
content = content.replace('tenant_metrics is not None and not isinstance(tenant_metrics, dict)', 'tenant_metrics is not None')

# Fix empty condition
content = re.sub(r'if (correlation_matrices|granger_matrices|te_matrices):', r'if \1 is not None:', content)

# Remove problematic isna
content = content.replace('if attack_vs_baseline_col in stats_df.columns and not pd.isna(row.get(attack_vs_baseline_col)):', 
                          'if attack_vs_baseline_col in stats_df.columns and row.get(attack_vs_baseline_col) is not None:')

with open('src/insight_aggregation.py', 'w') as file:
    file.write(content)

print("Fixed insight_aggregation.py")
