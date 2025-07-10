"""
Module: gpu_acceleration.py
Description: Utilities for GPU acceleration using CuPy, PyTorch, or TensorFlow.

This module provides functions and utilities to accelerate computationally intensive 
calculations using GPU when available, with fallback to CPU when necessary.
Supports acceleration of operations like correlation, spectral decomposition, and matrix calculations.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import time

# Setup logging
logger = logging.getLogger(__name__)

# Global flag to control GPU acceleration
GPU_ENABLED = False
ACCELERATION_BACKEND = None # Can be 'cupy', 'pytorch', 'tensorflow'

def check_gpu_availability() -> bool:
    """
    Checks if a GPU is available for computation acceleration.
    
    Returns:
        bool: True if GPU is available, False otherwise
    """
    global GPU_ENABLED, ACCELERATION_BACKEND
    
    # If already checked, return cached result
    if GPU_ENABLED is not None:
        return GPU_ENABLED
    
    # Try importing backends in order of preference
    GPU_ENABLED = False
    
    # 1. Try CuPy (preferred for NumPy compatibility)
    try:
        import cupy as cp
        # Test if it works
        x = cp.array([1, 2, 3])
        y = x * 2
        result = y.sum().get()
        
        if result == 12:  # Basic check
            GPU_ENABLED = True
            ACCELERATION_BACKEND = "cupy"
            logger.info("GPU available via CuPy")
            return True
    except (ImportError, Exception) as e:
        logger.debug(f"CuPy not available: {str(e)}")
    
    # 2. Try PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            GPU_ENABLED = True
            ACCELERATION_BACKEND = "pytorch"
            logger.info("GPU available via PyTorch")
            return True
    except ImportError:
        logger.debug("PyTorch not available")
    
    # 3. Try TensorFlow
    try:
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            GPU_ENABLED = True
            ACCELERATION_BACKEND = "tensorflow"
            logger.info("GPU available via TensorFlow")
            return True
    except ImportError:
        logger.debug("TensorFlow not available")
    
    logger.info("No GPU available. Using CPU computation.")
    return False

def get_backend():
    """
    Returns the current GPU acceleration backend.
    """
    return ACCELERATION_BACKEND if GPU_ENABLED else 'numpy'

def to_gpu(data: Union[np.ndarray, pd.DataFrame]) -> Any:
    """
    Moves a NumPy array or DataFrame to the GPU.
    
    Args:
        data: NumPy array or Pandas DataFrame to transfer
        
    Returns:
        Corresponding object in the GPU backend
    """
    if not GPU_ENABLED:
        return data
    
    if ACCELERATION_BACKEND == "cupy":
        import cupy as cp
        if isinstance(data, np.ndarray):
            return cp.asarray(data)
        elif isinstance(data, pd.DataFrame):
            # For DataFrames, transfer only numeric values
            return {col: cp.asarray(data[col].values) for col in data.select_dtypes(include=['number']).columns}
    
    elif ACCELERATION_BACKEND == "pytorch":
        import torch
        if isinstance(data, np.ndarray):
            return torch.tensor(data, device='cuda')
        elif isinstance(data, pd.DataFrame):
            return {col: torch.tensor(data[col].values, device='cuda') 
                   for col in data.select_dtypes(include=['number']).columns}
    
    elif ACCELERATION_BACKEND == "tensorflow":
        import tensorflow as tf
        if isinstance(data, np.ndarray):
            return tf.convert_to_tensor(data, dtype=tf.float32)
        elif isinstance(data, pd.DataFrame):
            return {col: tf.convert_to_tensor(data[col].values, dtype=tf.float32) 
                   for col in data.select_dtypes(include=['number']).columns}
    
    return data

def to_cpu(gpu_data: Any) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    Moves GPU data back to the CPU.
    
    Args:
        gpu_data: Data on the GPU to transfer
        
    Returns:
        np.ndarray or dictionary of NumPy arrays
    """
    if not GPU_ENABLED:
        return gpu_data
    
    if ACCELERATION_BACKEND == "cupy":
        import cupy as cp
        if isinstance(gpu_data, cp.ndarray):
            return gpu_data.get()
        elif isinstance(gpu_data, dict):
            return {key: val.get() if isinstance(val, cp.ndarray) else val 
                    for key, val in gpu_data.items()}
    
    elif ACCELERATION_BACKEND == "pytorch":
        import torch
        if isinstance(gpu_data, torch.Tensor):
            return gpu_data.cpu().numpy()
        elif isinstance(gpu_data, dict):
            return {key: val.cpu().numpy() if isinstance(val, torch.Tensor) else val 
                    for key, val in gpu_data.items()}
    
    elif ACCELERATION_BACKEND == "tensorflow":
        import tensorflow as tf
        if isinstance(gpu_data, tf.Tensor):
            return gpu_data.numpy()
        elif isinstance(gpu_data, dict):
            return {key: val.numpy() if isinstance(val, tf.Tensor) else val 
                    for key, val in gpu_data.items()}
    
    return gpu_data

def configure_gpu(enable: bool = True, backend: str = 'cupy'):
    """
    Configures and enables GPU acceleration.
    
    Args:
        enable: Whether to enable GPU acceleration
        backend: The backend to use for acceleration ('cupy', 'pytorch', 'tensorflow')
    """
    global GPU_ENABLED, ACCELERATION_BACKEND
    
    if not enable:
        GPU_ENABLED = False
        ACCELERATION_BACKEND = None
        logger.info("GPU acceleration is disabled.")
        return

    try:
        if backend == 'cupy':
            import cupy as cp
            cp.cuda.runtime.getDeviceCount()
            ACCELERATION_BACKEND = 'cupy'
        elif backend == 'pytorch':
            import torch
            if not torch.cuda.is_available():
                raise ImportError("PyTorch is installed, but CUDA is not available.")
            ACCELERATION_BACKEND = 'pytorch'
        elif backend == 'tensorflow':
            import tensorflow as tf
            if not tf.config.list_physical_devices('GPU'):
                raise ImportError("TensorFlow is installed, but no GPU is available.")
            ACCELERATION_BACKEND = 'tensorflow'
        else:
            raise ValueError(f"Unsupported backend: {backend}")
            
        GPU_ENABLED = True
        logger.info(f"GPU acceleration enabled successfully with backend: {ACCELERATION_BACKEND}")

    except ImportError as e:
        logger.warning(f"Failed to import {backend}. GPU acceleration will be disabled. Error: {e}")
        GPU_ENABLED = False
        ACCELERATION_BACKEND = None
    except Exception as e:
        logger.error(f"An error occurred during GPU configuration with backend {backend}. GPU acceleration disabled. Error: {e}")
        GPU_ENABLED = False
        ACCELERATION_BACKEND = None

def calculate_correlation_matrix_gpu(data: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
    """
    Calculates the correlation matrix using GPU when available.
    
    Args:
        data: DataFrame with the data for correlation
        method: Correlation method ('pearson', 'spearman', 'kendall')
        
    Returns:
        pd.DataFrame: Correlation matrix
    """
    start_time = time.time()
    
    if not GPU_ENABLED or method != 'pearson':
        # Fallback to CPU if GPU is not available or method is not Pearson
        result = data.corr(method=method)
        logger.debug(f"Correlation calculated on CPU in {time.time() - start_time:.2f}s")
        return result
    
    # Use GPU for Pearson correlation
    if ACCELERATION_BACKEND == "cupy":
        import cupy as cp
        
        # Select only numeric columns
        numeric_data = data.select_dtypes(include=['number'])
        
        # Transfer to GPU
        gpu_data = cp.asarray(numeric_data.values)
        
        # Normalize (necessary for Pearson correlation)
        gpu_data = (gpu_data - cp.mean(gpu_data, axis=0)) / cp.std(gpu_data, axis=0, ddof=1)
        
        # Calculate correlation
        n = gpu_data.shape[0]
        corr_matrix = cp.dot(gpu_data.T, gpu_data) / (n - 1)
        
        # Transfer back to CPU
        cpu_corr = corr_matrix.get()
        
        result = pd.DataFrame(
            cpu_corr, 
            index=numeric_data.columns,
            columns=numeric_data.columns
        )
        
    elif ACCELERATION_BACKEND == "pytorch":
        import torch
        
        numeric_data = data.select_dtypes(include=['number'])
        gpu_data = torch.tensor(numeric_data.values, dtype=torch.float32, device='cuda')
        
        # Normalize
        mean = torch.mean(gpu_data, dim=0)
        std = torch.std(gpu_data, dim=0, unbiased=True)
        gpu_data = (gpu_data - mean) / std
        
        # Calculate correlation
        n = gpu_data.shape[0]
        corr_matrix = torch.matmul(gpu_data.T, gpu_data) / (n - 1)
        
        # Transfer to CPU
        cpu_corr = corr_matrix.cpu().numpy()
        
        result = pd.DataFrame(
            cpu_corr, 
            index=numeric_data.columns,
            columns=numeric_data.columns
        )
        
    elif ACCELERATION_BACKEND == "tensorflow":
        import tensorflow as tf
        
        numeric_data = data.select_dtypes(include=['number'])
        gpu_data = tf.convert_to_tensor(numeric_data.values, dtype=tf.float32)
        
        # Normalize
        mean = tf.reduce_mean(gpu_data, axis=0)
        std = tf.math.reduce_std(gpu_data, axis=0)
        gpu_data = (gpu_data - mean) / std
        
        # Calculate correlation
        n = tf.shape(gpu_data)[0]
        corr_matrix = tf.matmul(gpu_data, gpu_data, transpose_a=True) / tf.cast(n - 1, tf.float32)
        
        # Transfer to CPU
        cpu_corr = corr_matrix.numpy()
        
        result = pd.DataFrame(
            cpu_corr, 
            index=numeric_data.columns,
            columns=numeric_data.columns
        )
    
    logger.info(f"Correlation calculated on GPU ({ACCELERATION_BACKEND}) in {time.time() - start_time:.2f}s")
    return result

def process_large_dataframe_chunks_gpu(
    df: pd.DataFrame,
    process_fn: Callable,
    chunk_size: int = 5000,
    **kwargs
) -> pd.DataFrame:
    """
    Processes a large DataFrame in chunks using GPU when available.
    
    Args:
        df: DataFrame to process
        process_fn: Function to process each chunk
        chunk_size: Size of each chunk
        **kwargs: Additional arguments for process_fn
        
    Returns:
        pd.DataFrame: Concatenated results
    """
    if len(df) <= chunk_size:
        return process_fn(df, **kwargs)
    
    # Split into chunks
    chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    results = []
    
    for i, chunk in enumerate(chunks):
        logger.debug(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk)} rows)")
        chunk_result = process_fn(chunk, **kwargs)
        results.append(chunk_result)
    
    # Concatenate results
    if isinstance(results[0], pd.DataFrame):
        return pd.concat(results)
    else:
        # For other return types, adjust accordingly
        return results

# --- Backend-Specific Implementations ---

# Example of a backend-specific function: cohens_d
# Note: This is a simplified example. A real implementation would need more robust error handling.

def cohens_d_gpu(group1, group2, backend: str) -> float:
    """
    Calculates Cohen's d on the GPU.
    """
    if not GPU_ENABLED or backend != ACCELERATION_BACKEND:
        raise RuntimeError(f"GPU backend '{backend}' is not configured or enabled.")

    # Move data to GPU
    g1_gpu = to_gpu(group1, backend)
    g2_gpu = to_gpu(group2, backend)

    if backend == 'cupy':
        import cupy as cp
        n1, n2 = len(g1_gpu), len(g2_gpu)
        if n1 == 0 or n2 == 0 or (n1 + n2 - 2) == 0: return 0.0
        s1, s2 = cp.var(g1_gpu, ddof=1), cp.var(g2_gpu, ddof=1)
        pooled_std = cp.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
        if pooled_std == 0: return 0.0
        u1, u2 = cp.mean(g1_gpu), cp.mean(g2_gpu)
        d = (u1 - u2) / pooled_std
        return float(to_cpu(d))

    elif backend == 'pytorch':
        import torch
        n1, n2 = g1_gpu.shape[0], g2_gpu.shape[0]
        if n1 == 0 or n2 == 0 or (n1 + n2 - 2) == 0: return 0.0
        s1, s2 = torch.var(g1_gpu, unbiased=True), torch.var(g2_gpu, unbiased=True)
        pooled_std = torch.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
        if pooled_std == 0: return 0.0
        u1, u2 = torch.mean(g1_gpu), torch.mean(g2_gpu)
        d = (u1 - u2) / pooled_std
        return float(to_cpu(d))

    elif backend == 'tensorflow':
        import tensorflow as tf
        n1, n2 = tf.shape(g1_gpu)[0], tf.shape(g2_gpu)[0]
        if n1 == 0 or n2 == 0 or (n1 + n2 - 2) == 0: return 0.0
        s1, s2 = tf.math.reduce_variance(g1_gpu), tf.math.reduce_variance(g2_gpu) # Note: TF variance is population based by default
        # This part would need a more careful implementation for unbiased variance in TF
        pooled_std = tf.sqrt(((tf.cast(n1 - 1, tf.float32) * s1) + (tf.cast(n2 - 1, tf.float32) * s2)) / tf.cast(n1 + n2 - 2, tf.float32))
        if pooled_std == 0: return 0.0
        u1, u2 = tf.reduce_mean(g1_gpu), tf.reduce_mean(g2_gpu)
        d = (u1 - u2) / pooled_std
        return float(to_cpu(d))
        
    raise NotImplementedError(f"Cohen's d not implemented for backend: {backend}")
