"""
Module: gpu_acceleration.py
Description: Módulo para aceleração de cálculos usando GPU.

Este módulo fornece funções e utilitários para acelerar cálculos computacionalmente 
intensivos usando GPU quando disponível, com fallback para CPU quando necessário.
Suporta aceleração de operações como correlação, decomposição espectral e cálculos matriciais.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import time

# Configuração de logging
logger = logging.getLogger(__name__)

# Variáveis globais para controlar o estado da aceleração GPU
_GPU_AVAILABLE = None
_GPU_BACKEND = None

def check_gpu_availability() -> bool:
    """
    Verifica se há GPU disponível para aceleração de cálculos.
    
    Returns:
        bool: True se GPU está disponível, False caso contrário
    """
    global _GPU_AVAILABLE, _GPU_BACKEND
    
    # Se já verificamos, retorna o resultado em cache
    if _GPU_AVAILABLE is not None:
        return _GPU_AVAILABLE
    
    # Tenta importar backends em ordem de preferência
    _GPU_AVAILABLE = False
    
    # 1. Tenta CuPy (preferido para compatibilidade NumPy)
    try:
        import cupy as cp
        # Testa se realmente funciona
        x = cp.array([1, 2, 3])
        y = x * 2
        result = y.sum().get()
        
        if result == 12:  # Verificação básica
            _GPU_AVAILABLE = True
            _GPU_BACKEND = "cupy"
            logger.info("GPU disponível via CuPy")
            return True
    except (ImportError, Exception) as e:
        logger.debug(f"CuPy não disponível: {str(e)}")
    
    # 2. Tenta PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            _GPU_AVAILABLE = True
            _GPU_BACKEND = "torch"
            logger.info("GPU disponível via PyTorch")
            return True
    except ImportError:
        logger.debug("PyTorch não disponível")
    
    # 3. Tenta TensorFlow
    try:
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            _GPU_AVAILABLE = True
            _GPU_BACKEND = "tensorflow"
            logger.info("GPU disponível via TensorFlow")
            return True
    except ImportError:
        logger.debug("TensorFlow não disponível")
    
    logger.info("Nenhuma GPU disponível. Usando computação CPU.")
    return False

def get_gpu_backend() -> str:
    """
    Retorna o backend GPU em uso.
    
    Returns:
        str: Nome do backend ('cupy', 'torch', 'tensorflow', ou 'none')
    """
    if not check_gpu_availability():
        return "none"
    return _GPU_BACKEND

def to_gpu(data: Union[np.ndarray, pd.DataFrame]) -> Any:
    """
    Transfere dados para a GPU.
    
    Args:
        data: Array NumPy ou DataFrame Pandas para transferir
        
    Returns:
        Objeto correspondente no backend GPU
    """
    if not check_gpu_availability():
        return data
    
    if _GPU_BACKEND == "cupy":
        import cupy as cp
        if isinstance(data, np.ndarray):
            return cp.asarray(data)
        elif isinstance(data, pd.DataFrame):
            # Para DataFrames, transferimos apenas os valores numéricos
            return {col: cp.asarray(data[col].values) for col in data.select_dtypes(include=['number']).columns}
    
    elif _GPU_BACKEND == "torch":
        import torch
        if isinstance(data, np.ndarray):
            return torch.tensor(data, device='cuda')
        elif isinstance(data, pd.DataFrame):
            return {col: torch.tensor(data[col].values, device='cuda') 
                   for col in data.select_dtypes(include=['number']).columns}
    
    elif _GPU_BACKEND == "tensorflow":
        import tensorflow as tf
        if isinstance(data, np.ndarray):
            return tf.convert_to_tensor(data, dtype=tf.float32)
        elif isinstance(data, pd.DataFrame):
            return {col: tf.convert_to_tensor(data[col].values, dtype=tf.float32) 
                   for col in data.select_dtypes(include=['number']).columns}
    
    return data

def to_cpu(gpu_data: Any) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    Transfere dados da GPU de volta para a CPU.
    
    Args:
        gpu_data: Dados na GPU para transferir
        
    Returns:
        np.ndarray ou dicionário de arrays NumPy
    """
    if not check_gpu_availability():
        return gpu_data
    
    if _GPU_BACKEND == "cupy":
        import cupy as cp
        if isinstance(gpu_data, cp.ndarray):
            return gpu_data.get()
        elif isinstance(gpu_data, dict):
            return {key: val.get() if isinstance(val, cp.ndarray) else val 
                    for key, val in gpu_data.items()}
    
    elif _GPU_BACKEND == "torch":
        import torch
        if isinstance(gpu_data, torch.Tensor):
            return gpu_data.cpu().numpy()
        elif isinstance(gpu_data, dict):
            return {key: val.cpu().numpy() if isinstance(val, torch.Tensor) else val 
                    for key, val in gpu_data.items()}
    
    elif _GPU_BACKEND == "tensorflow":
        import tensorflow as tf
        if isinstance(gpu_data, tf.Tensor):
            return gpu_data.numpy()
        elif isinstance(gpu_data, dict):
            return {key: val.numpy() if isinstance(val, tf.Tensor) else val 
                    for key, val in gpu_data.items()}
    
    return gpu_data

def calculate_correlation_matrix_gpu(data: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
    """
    Calcula a matriz de correlação usando GPU quando disponível.
    
    Args:
        data: DataFrame com os dados para correlação
        method: Método de correlação ('pearson', 'spearman', 'kendall')
        
    Returns:
        pd.DataFrame: Matriz de correlação
    """
    start_time = time.time()
    
    if not check_gpu_availability() or method != 'pearson':
        # Fallback para CPU se GPU não estiver disponível ou método não for Pearson
        result = data.corr(method=method)
        logger.debug(f"Correlação calculada na CPU em {time.time() - start_time:.2f}s")
        return result
    
    # Usar GPU para correlação de Pearson
    if _GPU_BACKEND == "cupy":
        import cupy as cp
        
        # Selecionar apenas colunas numéricas
        numeric_data = data.select_dtypes(include=['number'])
        
        # Transferir para GPU
        gpu_data = cp.asarray(numeric_data.values)
        
        # Normalizar (necessário para correlação de Pearson)
        gpu_data = (gpu_data - cp.mean(gpu_data, axis=0)) / cp.std(gpu_data, axis=0, ddof=1)
        
        # Calcular correlação
        n = gpu_data.shape[0]
        corr_matrix = cp.dot(gpu_data.T, gpu_data) / (n - 1)
        
        # Transferir de volta para CPU
        cpu_corr = corr_matrix.get()
        
        result = pd.DataFrame(
            cpu_corr, 
            index=numeric_data.columns,
            columns=numeric_data.columns
        )
        
    elif _GPU_BACKEND == "torch":
        import torch
        
        numeric_data = data.select_dtypes(include=['number'])
        gpu_data = torch.tensor(numeric_data.values, dtype=torch.float32, device='cuda')
        
        # Normalizar
        mean = torch.mean(gpu_data, dim=0)
        std = torch.std(gpu_data, dim=0, unbiased=True)
        gpu_data = (gpu_data - mean) / std
        
        # Calcular correlação
        n = gpu_data.shape[0]
        corr_matrix = torch.matmul(gpu_data.T, gpu_data) / (n - 1)
        
        # Transferir para CPU
        cpu_corr = corr_matrix.cpu().numpy()
        
        result = pd.DataFrame(
            cpu_corr, 
            index=numeric_data.columns,
            columns=numeric_data.columns
        )
        
    elif _GPU_BACKEND == "tensorflow":
        import tensorflow as tf
        
        numeric_data = data.select_dtypes(include=['number'])
        gpu_data = tf.convert_to_tensor(numeric_data.values, dtype=tf.float32)
        
        # Normalizar
        mean = tf.reduce_mean(gpu_data, axis=0)
        std = tf.math.reduce_std(gpu_data, axis=0)
        gpu_data = (gpu_data - mean) / std
        
        # Calcular correlação
        n = tf.shape(gpu_data)[0]
        corr_matrix = tf.matmul(gpu_data, gpu_data, transpose_a=True) / tf.cast(n - 1, tf.float32)
        
        # Transferir para CPU
        cpu_corr = corr_matrix.numpy()
        
        result = pd.DataFrame(
            cpu_corr, 
            index=numeric_data.columns,
            columns=numeric_data.columns
        )
    
    logger.info(f"Correlação calculada na GPU ({_GPU_BACKEND}) em {time.time() - start_time:.2f}s")
    return result

def process_large_dataframe_chunks_gpu(
    df: pd.DataFrame,
    process_fn: Callable,
    chunk_size: int = 5000,
    **kwargs
) -> pd.DataFrame:
    """
    Processa um DataFrame grande em chunks usando GPU quando disponível.
    
    Args:
        df: DataFrame para processar
        process_fn: Função para processar cada chunk
        chunk_size: Tamanho de cada chunk
        **kwargs: Argumentos adicionais para process_fn
        
    Returns:
        pd.DataFrame: Resultados concatenados
    """
    if len(df) <= chunk_size:
        return process_fn(df, **kwargs)
    
    # Dividir em chunks
    chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    results = []
    
    for i, chunk in enumerate(chunks):
        logger.debug(f"Processando chunk {i+1}/{len(chunks)} ({len(chunk)} linhas)")
        chunk_result = process_fn(chunk, **kwargs)
        results.append(chunk_result)
    
    # Concatenar resultados
    if isinstance(results[0], pd.DataFrame):
        return pd.concat(results)
    else:
        # Para outros tipos de retorno, ajustar de acordo
        return results

# Funções específicas para operações comuns

def fast_pca_gpu(data: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Executa PCA acelerado por GPU quando disponível.
    
    Args:
        data: Array NumPy com os dados
        n_components: Número de componentes para extrair
        
    Returns:
        Tuple: (componentes, variância explicada, dados transformados)
    """
    if not check_gpu_availability():
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(data)
        return pca.components_, pca.explained_variance_ratio_, transformed
    
    if _GPU_BACKEND == "cupy":
        import cupy as cp
        
        # Transferir para GPU
        gpu_data = cp.asarray(data)
        
        # Centralizar os dados
        gpu_data = gpu_data - cp.mean(gpu_data, axis=0)
        
        # SVD
        U, S, V = cp.linalg.svd(gpu_data, full_matrices=False)
        
        # Variância explicada
        explained_variance = (S ** 2) / (gpu_data.shape[0] - 1)
        total_var = explained_variance.sum()
        explained_variance_ratio = explained_variance / total_var
        
        # Componentes e transformação
        components = V[:n_components]
        transformed = gpu_data @ components.T
        
        # Transferir de volta para CPU
        return components.get(), explained_variance_ratio[:n_components].get(), transformed.get()
    
    # Implementações para torch e tensorflow seriam similares...
    # Por simplicidade, fallback para sklearn em outros casos
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(data)
    return pca.components_, pca.explained_variance_ratio_, transformed
