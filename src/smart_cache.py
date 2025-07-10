"""
Module: smart_cache.py
Description: A smart caching mechanism to store and retrieve intermediate results.

This module implements an intelligent caching system that identifies when
data or settings have changed and only reprocesses the necessary parts, significantly
increasing performance for large or repeated analyses.
"""

import os
import logging
import json
import hashlib
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import pickle
from pathlib import Path
import time
import datetime
import shutil

logger = logging.getLogger(__name__)

class SmartCache:
    """
    A simple file-based cache that stores results of operations based on a key.
    The key is typically a hash of the inputs that generated the result.
    """
    
    def __init__(
        self, 
        cache_dir: str = "./cache",
        max_age_days: float = 7.0,
        auto_cleanup: bool = True,
        compression: str = "gzip"
    ):
        """
        Initializes the cache.
        
        Args:
            cache_dir (str): The directory where cache files will be stored.
            max_age_days: Maximum age of the cache in days before being considered obsolete
            auto_cleanup: If True, automatically cleans up old caches
            compression: Compression method for cache files
        """
        self.cache_dir = Path(cache_dir)
        self.max_age_seconds = max_age_days * 24 * 3600
        self.auto_cleanup = auto_cleanup
        self.compression = compression
        self._ensure_cache_dir()
        
        # Register cache usage statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'saved_time': 0.0,
            'saved_operations': 0
        }
        
        logger.info(f"Smart cache system initialized at {self.cache_dir}")
        
        # Perform automatic cleanup if configured
        if self.auto_cleanup:
            self.cleanup_old_caches()
    
    def _ensure_cache_dir(self):
        """Ensures that the cache directory exists."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create index file if it doesn't exist
        index_file = self.cache_dir / "cache_index.json"
        if not index_file.exists():
            with open(index_file, 'w') as f:
                json.dump({
                    "created_at": datetime.datetime.now().isoformat(),
                    "entries": {}
                }, f)
    
    def _compute_hash(self, data: Any) -> str:
        """
        Computes a hash for the provided data.
        
        Args:
            data: The data to compute the hash for
            
        Returns:
            str: Hexadecimal string of the SHA-256 hash
        """
        if isinstance(data, pd.DataFrame):
            # For DataFrames, use an approach that considers structure and values
            buffer = []
            buffer.append(str(data.shape))
            buffer.append(str(data.columns.tolist()))
            buffer.append(str(data.index.tolist()[:100]))  # Use only part of the index for performance
            buffer.append(str(data.dtypes.to_dict()))
            
            # Add a sample of values for the hash
            if not data.empty:
                sample_size = min(1000, len(data))
                sample = data.sample(sample_size) if len(data) > sample_size else data
                buffer.append(pd.util.hash_pandas_object(sample).sum())
            
            data_str = "".join([str(item) for item in buffer])
        elif isinstance(data, dict):
            # Sort keys to ensure consistency
            data_str = json.dumps(data, sort_keys=True, default=str)
        elif isinstance(data, (list, tuple)):
            # Convert each item to string for lists
            data_str = json.dumps([str(item) for item in data], default=str)
        else:
            # For other types, use string representation
            data_str = str(data)
        
        # Calculate SHA-256 hash
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str, subfolder: str = None) -> Path:
        """
        Gets the full path for a cache file based on the key.
        
        Args:
            cache_key: Cache key (hash)
            subfolder: Optional, subfolder for organization
            
        Returns:
            Path: Full path to the cache file
        """
        if subfolder:
            folder = self.cache_dir / subfolder
            folder.mkdir(exist_ok=True)
            return folder / f"{cache_key}.cache"
        else:
            return self.cache_dir / f"{cache_key}.cache"
    
    def _update_index(self, cache_key: str, metadata: Dict[str, Any]):
        """
        Updates the index file with information about the cache.
        
        Args:
            cache_key: Cache key
            metadata: Metadata about the cache (type, size, etc.)
        """
        index_file = self.cache_dir / "cache_index.json"
        
        try:
            with open(index_file, 'r') as f:
                index = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            # Recreate index if corrupted or nonexistent
            index = {
                "created_at": datetime.datetime.now().isoformat(),
                "entries": {}
            }
        
        # Add or update entry
        metadata['last_access'] = datetime.datetime.now().isoformat()
        index["entries"][cache_key] = metadata
        
        # Save updated index
        with open(index_file, 'w') as f:
            json.dump(index, f, default=str)
    
    def cached_operation(
        self,
        operation: Callable,
        operation_args: tuple = (),
        operation_kwargs: Dict[str, Any] = {},
        dependencies: Dict[str, Any] = {},
        cache_name: str = None,
        subfolder: str = None,
        force_recompute: bool = False
    ) -> Any:
        """
        Executes an operation with intelligent caching. If the result is already cached
        and dependencies haven't changed, returns the cached result. Otherwise, executes
        the operation and stores the result in the cache.
        
        Args:
            operation: Function or method to be executed
            operation_args: Positional arguments for the operation
            operation_kwargs: Named arguments for the operation
            dependencies: Dictionary with values that, when changed, invalidate the cache
            cache_name: Optional name for the cache (if None, uses the function name)
            subfolder: Optional subfolder within the cache directory
            force_recompute: If True, forces re-execution even if a cache exists
            
        Returns:
            Any: Result of the operation (from cache or newly calculated)
        """
        if operation is None:
            raise ValueError("The operation cannot be None")
        
        # Generate a cache key based on the function and dependencies
        fn_name = operation.__name__ if hasattr(operation, '__name__') else str(operation)
        cache_name = cache_name or fn_name
        
        # Create combined hash of function, arguments, and dependencies
        hash_components = {
            'function': fn_name,
            'args': operation_args,
            'kwargs': operation_kwargs,
            'dependencies': dependencies
        }
        
        cache_key = f"{cache_name}_{self._compute_hash(hash_components)}"
        cache_path = self._get_cache_path(cache_key, subfolder)
        
        # Check if the cache exists and is valid
        if cache_path.exists() and not force_recompute:
            try:
                # Load cache metadata
                cache_mtime = cache_path.stat().st_mtime
                current_time = time.time()
                cache_age = current_time - cache_mtime
                
                # Check if the cache is not too old
                if cache_age <= self.max_age_seconds:
                    # Load the result from the cache
                    with open(cache_path, 'rb') as f:
                        start_time = time.time()
                        cached_result = pickle.load(f)
                        load_time = time.time() - start_time
                    
                    # Update usage statistics
                    self.stats['hits'] += 1
                    self.stats['saved_time'] += cached_result.get('original_execution_time', 0)
                    self.stats['saved_operations'] += 1
                    
                    # Update the index with recent access
                    self._update_index(cache_key, {
                        'name': cache_name,
                        'last_access': datetime.datetime.now().isoformat(),
                        'data_type': str(type(cached_result['result'])),
                        'size_bytes': cache_path.stat().st_size,
                        'is_valid': True
                    })
                    
                    logger.info(f"Cache hit for '{cache_name}' (age: {cache_age:.1f}s, loaded in {load_time:.3f}s)")
                    return cached_result['result']
                else:
                    logger.info(f"Cache expired for '{cache_name}' (age: {cache_age:.1f}s > {self.max_age_seconds}s)")
            except Exception as e:
                logger.warning(f"Error loading cache '{cache_name}': {e}")
        
        # If we reached here, no valid cache was found or force_recompute is True
        self.stats['misses'] += 1
        
        # Execute the operation and measure time
        logger.info(f"Executing operation '{cache_name}' (no valid cache available)")
        start_time = time.time()
        result = operation(*operation_args, **operation_kwargs)
        execution_time = time.time() - start_time
        
        # Save the result in the cache
        try:
            cache_data = {
                'result': result,
                'cached_at': datetime.datetime.now().isoformat(),
                'original_execution_time': execution_time,
                'function': fn_name,
                'dependencies_hash': self._compute_hash(dependencies)
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Update the index
            self._update_index(cache_key, {
                'name': cache_name,
                'cached_at': datetime.datetime.now().isoformat(),
                'last_access': datetime.datetime.now().isoformat(),
                'execution_time': execution_time,
                'data_type': str(type(result)),
                'size_bytes': cache_path.stat().st_size,
                'is_valid': True
            })
            
            logger.info(f"Operation '{cache_name}' executed in {execution_time:.3f}s and saved to cache")
        except Exception as e:
            logger.error(f"Error saving cache for '{cache_name}': {e}")
        
        return result
    
    def clear_cache(self, subfolder: str = None):
        """
        Clears all cache files.
        
        Args:
            subfolder: If provided, only clears this subfolder
        """
        target_dir = self.cache_dir / subfolder if subfolder else self.cache_dir
        
        if not target_dir.exists():
            logger.warning(f"Cache directory {target_dir} does not exist")
            return
        
        # Delete all cache files, but preserve the directory and index
        count = 0
        total_size = 0
        
        for cache_file in target_dir.glob("*.cache"):
            try:
                file_size = cache_file.stat().st_size
                cache_file.unlink()
                count += 1
                total_size += file_size
            except Exception as e:
                logger.error(f"Error deleting cache file {cache_file}: {e}")
        
        # Recreate the index if we are clearing everything
        if not subfolder:
            index_file = self.cache_dir / "cache_index.json"
            with open(index_file, 'w') as f:
                json.dump({
                    "created_at": datetime.datetime.now().isoformat(),
                    "entries": {}
                }, f)
        
        logger.info(f"Cache cleared: {count} files removed, {total_size / (1024*1024):.2f} MB freed")
    
    def cleanup_old_caches(self, min_age_days: float = None):
        """
        Removes caches older than the age limit.
        
        Args:
            min_age_days: Minimum age in days to remove (if None, uses max_age_days)
        """
        min_age_seconds = min_age_days * 24 * 3600 if min_age_days else self.max_age_seconds
        current_time = time.time()
        
        count = 0
        total_size = 0
        
        # Check all cache files
        for cache_file in self.cache_dir.glob("**/*.cache"):
            try:
                mtime = cache_file.stat().st_mtime
                age = current_time - mtime
                
                if age > min_age_seconds:
                    file_size = cache_file.stat().st_size
                    cache_file.unlink()
                    count += 1
                    total_size += file_size
                    logger.debug(f"Removed old cache: {cache_file} (age: {age / 86400:.1f} days)")
            except Exception as e:
                logger.error(f"Error processing cache file {cache_file}: {e}")
        
        logger.info(f"Cache cleanup: {count} old files removed, {total_size / (1024*1024):.2f} MB freed")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Gets statistics about cache usage.
        
        Returns:
            Dict: Cache statistics
        """
        # Basic hit/miss statistics
        stats = self.stats.copy()
        
        # Add storage statistics
        total_size = 0
        file_count = 0
        
        for cache_file in self.cache_dir.glob("**/*.cache"):
            total_size += cache_file.stat().st_size
            file_count += 1
        
        stats['file_count'] = file_count
        stats['total_size_mb'] = total_size / (1024 * 1024)
        stats['hit_rate'] = stats['hits'] / (stats['hits'] + stats['misses']) if (stats['hits'] + stats['misses']) > 0 else 0
        
        return stats
    
    def get_cache_info(self, cache_key: str = None) -> Dict[str, Any]:
        """
        Gets information about a specific cache or all caches.
        
        Args:
            cache_key: Specific cache key or None for all
            
        Returns:
            Dict: Cache information
        """
        index_file = self.cache_dir / "cache_index.json"
        
        if not index_file.exists():
            return {"error": "Cache index not found"}
        
        try:
            with open(index_file, 'r') as f:
                index = json.load(f)
            
            if cache_key:
                return index["entries"].get(cache_key, {"error": "Cache not found"})
            else:
                return index
        except Exception as e:
            return {"error": f"Error reading cache index: {e}"}

def cached_function(
    cache_instance: SmartCache, 
    dependencies: Dict[str, Any] = None, 
    cache_name: str = None,
    subfolder: str = None,
    force_recompute: bool = False
):
    """
    Decorator for functions that should use the intelligent cache system.
    
    Args:
        cache_instance: Instance of SmartCache
        dependencies: Values that, when changed, invalidate the cache
        cache_name: Optional name for the cache
        subfolder: Optional subfolder within the cache directory
        force_recompute: If True, forces recomputation
        
    Returns:
        Callable: Decorated function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            return cache_instance.cached_operation(
                operation=func,
                operation_args=args,
                operation_kwargs=kwargs,
                dependencies=dependencies or {},
                cache_name=cache_name or func.__name__,
                subfolder=subfolder,
                force_recompute=force_recompute
            )
        return wrapper
    return decorator
