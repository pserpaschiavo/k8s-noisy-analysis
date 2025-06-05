#!/usr/bin/env python3
"""
Teste simplificado para verificar se temos alguma exceção com as funções modificadas.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Criar diretório para os testes
output_dir = "output_test"
os.makedirs(output_dir, exist_ok=True)

# Criar dados de teste simples
start_time = datetime.now()
data = []
for i in range(20):
    timestamp = start_time + timedelta(minutes=i*10)
    data.append({
        'timestamp': timestamp,
        'value': np.sin(i/3) * 10 + 50 + np.random.normal(0, 2)
    })

df = pd.DataFrame(data)

# Calcular tempo relativo
start = df['timestamp'].min()
relative_minutes = [(t - start).total_seconds() / 60 for t in df['timestamp']]

# Plotar dados
plt.figure(figsize=(10, 6))
plt.plot(relative_minutes, df['value'], 'o-', linewidth=2)
plt.xlabel('Minutos desde o início')
plt.ylabel('Valor')
plt.title('Teste com Tempo Relativo')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Salvar imagem
plt.savefig(os.path.join(output_dir, 'teste_simples.png'))
plt.close()

print("Teste concluído com sucesso!")
print(f"Imagem salva em {os.path.join(output_dir, 'teste_simples.png')}")
