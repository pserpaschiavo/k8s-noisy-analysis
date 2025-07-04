"""
Module: smart_cache.py
Description: Sistema avançado de cache para evitar reprocessamento desnecessário.

Este módulo implementa um sistema de cache inteligente que identifica quando
dados ou configurações mudaram e apenas reprocessa o necessário, aumentando
significativamente a performance para análises grandes ou repetidas.
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
    Classe que implementa um sistema de cache inteligente para evitar
    reprocessamento desnecessário de dados e análises.
    """
    
    def __init__(
        self, 
        cache_dir: str = "./cache",
        max_age_days: float = 7.0,
        auto_cleanup: bool = True,
        compression: str = "gzip"
    ):
        """
        Inicializa o sistema de cache inteligente.
        
        Args:
            cache_dir: Diretório para armazenar os arquivos de cache
            max_age_days: Idade máxima do cache em dias antes de ser considerado obsoleto
            auto_cleanup: Se True, limpa automaticamente caches antigos
            compression: Método de compressão para arquivos de cache
        """
        self.cache_dir = Path(cache_dir)
        self.max_age_seconds = max_age_days * 24 * 3600
        self.auto_cleanup = auto_cleanup
        self.compression = compression
        self._ensure_cache_dir()
        
        # Registrar estatísticas de uso do cache
        self.stats = {
            'hits': 0,
            'misses': 0,
            'saved_time': 0.0,
            'saved_operations': 0
        }
        
        logger.info(f"Sistema de cache inteligente inicializado em {self.cache_dir}")
        
        # Executar limpeza automática se configurado
        if self.auto_cleanup:
            self.cleanup_old_caches()
    
    def _ensure_cache_dir(self):
        """Garante que o diretório de cache existe."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Criar arquivo de índice se não existir
        index_file = self.cache_dir / "cache_index.json"
        if not index_file.exists():
            with open(index_file, 'w') as f:
                json.dump({
                    "created_at": datetime.datetime.now().isoformat(),
                    "entries": {}
                }, f)
    
    def _compute_hash(self, data: Any) -> str:
        """
        Computa um hash para os dados fornecidos.
        
        Args:
            data: Os dados para calcular o hash
            
        Returns:
            str: String hexadecimal do hash SHA-256
        """
        if isinstance(data, pd.DataFrame):
            # Para DataFrames, usamos uma abordagem que considera a estrutura e os valores
            buffer = []
            buffer.append(str(data.shape))
            buffer.append(str(data.columns.tolist()))
            buffer.append(str(data.index.tolist()[:100]))  # Usar só parte do índice para performance
            buffer.append(str(data.dtypes.to_dict()))
            
            # Adicionar uma amostra de valores para o hash
            if not data.empty:
                sample_size = min(1000, len(data))
                sample = data.sample(sample_size) if len(data) > sample_size else data
                buffer.append(pd.util.hash_pandas_object(sample).sum())
            
            data_str = "".join([str(item) for item in buffer])
        elif isinstance(data, dict):
            # Ordenar as chaves para garantir consistência
            data_str = json.dumps(data, sort_keys=True, default=str)
        elif isinstance(data, (list, tuple)):
            # Para listas, convertemos cada item para string
            data_str = json.dumps([str(item) for item in data], default=str)
        else:
            # Para outros tipos, usamos a representação string
            data_str = str(data)
        
        # Calcular o hash SHA-256
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str, subfolder: str = None) -> Path:
        """
        Obtém o caminho completo para um arquivo de cache com base na chave.
        
        Args:
            cache_key: Chave de cache (hash)
            subfolder: Opcional, subpasta para organização
            
        Returns:
            Path: Caminho completo para o arquivo de cache
        """
        if subfolder:
            folder = self.cache_dir / subfolder
            folder.mkdir(exist_ok=True)
            return folder / f"{cache_key}.cache"
        else:
            return self.cache_dir / f"{cache_key}.cache"
    
    def _update_index(self, cache_key: str, metadata: Dict[str, Any]):
        """
        Atualiza o arquivo de índice com informações sobre o cache.
        
        Args:
            cache_key: Chave de cache
            metadata: Metadados sobre o cache (tipo, tamanho, etc.)
        """
        index_file = self.cache_dir / "cache_index.json"
        
        try:
            with open(index_file, 'r') as f:
                index = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            # Recriar índice se corrompido ou inexistente
            index = {
                "created_at": datetime.datetime.now().isoformat(),
                "entries": {}
            }
        
        # Adicionar ou atualizar entrada
        metadata['last_access'] = datetime.datetime.now().isoformat()
        index["entries"][cache_key] = metadata
        
        # Salvar índice atualizado
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
        Executa uma operação com caching inteligente. Se o resultado já estiver em cache
        e as dependências não tiverem mudado, retorna do cache. Caso contrário, executa
        a operação e armazena o resultado em cache.
        
        Args:
            operation: Função ou método a ser executado
            operation_args: Argumentos posicionais para a operação
            operation_kwargs: Argumentos nomeados para a operação
            dependencies: Dicionário com valores que, quando alterados, invalidam o cache
            cache_name: Nome opcional para o cache (se None, usa o nome da função)
            subfolder: Subpasta opcional dentro do diretório de cache
            force_recompute: Se True, força a reexecução mesmo se houver cache
            
        Returns:
            Any: Resultado da operação (do cache ou recém-calculado)
        """
        if operation is None:
            raise ValueError("A operação não pode ser None")
        
        # Gerar uma chave de cache baseada na função e dependências
        fn_name = operation.__name__ if hasattr(operation, '__name__') else str(operation)
        cache_name = cache_name or fn_name
        
        # Criar hash combinado da função, argumentos e dependências
        hash_components = {
            'function': fn_name,
            'args': operation_args,
            'kwargs': operation_kwargs,
            'dependencies': dependencies
        }
        
        cache_key = f"{cache_name}_{self._compute_hash(hash_components)}"
        cache_path = self._get_cache_path(cache_key, subfolder)
        
        # Verificar se o cache existe e é válido
        if cache_path.exists() and not force_recompute:
            try:
                # Carregar metadados do cache
                cache_mtime = cache_path.stat().st_mtime
                current_time = time.time()
                cache_age = current_time - cache_mtime
                
                # Verificar se o cache não está muito velho
                if cache_age <= self.max_age_seconds:
                    # Carregar o resultado do cache
                    with open(cache_path, 'rb') as f:
                        start_time = time.time()
                        cached_result = pickle.load(f)
                        load_time = time.time() - start_time
                    
                    # Atualizar estatísticas de uso
                    self.stats['hits'] += 1
                    self.stats['saved_time'] += cached_result.get('original_execution_time', 0)
                    self.stats['saved_operations'] += 1
                    
                    # Atualizar o índice com o acesso recente
                    self._update_index(cache_key, {
                        'name': cache_name,
                        'last_access': datetime.datetime.now().isoformat(),
                        'data_type': str(type(cached_result['result'])),
                        'size_bytes': cache_path.stat().st_size,
                        'is_valid': True
                    })
                    
                    logger.info(f"Cache hit para '{cache_name}' (idade: {cache_age:.1f}s, carregado em {load_time:.3f}s)")
                    return cached_result['result']
                else:
                    logger.info(f"Cache expirado para '{cache_name}' (idade: {cache_age:.1f}s > {self.max_age_seconds}s)")
            except Exception as e:
                logger.warning(f"Erro ao carregar cache '{cache_name}': {e}")
        
        # Se chegou aqui, não encontrou cache válido ou force_recompute é True
        self.stats['misses'] += 1
        
        # Executar a operação e medir o tempo
        logger.info(f"Executando operação '{cache_name}' (sem cache válido disponível)")
        start_time = time.time()
        result = operation(*operation_args, **operation_kwargs)
        execution_time = time.time() - start_time
        
        # Salvar o resultado em cache
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
            
            # Atualizar o índice
            self._update_index(cache_key, {
                'name': cache_name,
                'cached_at': datetime.datetime.now().isoformat(),
                'last_access': datetime.datetime.now().isoformat(),
                'execution_time': execution_time,
                'data_type': str(type(result)),
                'size_bytes': cache_path.stat().st_size,
                'is_valid': True
            })
            
            logger.info(f"Operação '{cache_name}' executada em {execution_time:.3f}s e salva em cache")
        except Exception as e:
            logger.error(f"Erro ao salvar cache para '{cache_name}': {e}")
        
        return result
    
    def clear_cache(self, subfolder: str = None):
        """
        Limpa todos os arquivos de cache.
        
        Args:
            subfolder: Se fornecido, limpa apenas esta subpasta
        """
        target_dir = self.cache_dir / subfolder if subfolder else self.cache_dir
        
        if not target_dir.exists():
            logger.warning(f"Diretório de cache {target_dir} não existe")
            return
        
        # Apagar todos os arquivos de cache, mas preservar o diretório e o índice
        count = 0
        total_size = 0
        
        for cache_file in target_dir.glob("*.cache"):
            try:
                file_size = cache_file.stat().st_size
                cache_file.unlink()
                count += 1
                total_size += file_size
            except Exception as e:
                logger.error(f"Erro ao apagar arquivo de cache {cache_file}: {e}")
        
        # Recriar o índice se estivermos limpando tudo
        if not subfolder:
            index_file = self.cache_dir / "cache_index.json"
            with open(index_file, 'w') as f:
                json.dump({
                    "created_at": datetime.datetime.now().isoformat(),
                    "entries": {}
                }, f)
        
        logger.info(f"Cache limpo: {count} arquivos removidos, {total_size / (1024*1024):.2f} MB liberados")
    
    def cleanup_old_caches(self, min_age_days: float = None):
        """
        Remove caches mais antigos que o limite de idade.
        
        Args:
            min_age_days: Idade mínima em dias para remover (se None, usa max_age_days)
        """
        min_age_seconds = min_age_days * 24 * 3600 if min_age_days else self.max_age_seconds
        current_time = time.time()
        
        count = 0
        total_size = 0
        
        # Checar todos os arquivos de cache
        for cache_file in self.cache_dir.glob("**/*.cache"):
            try:
                mtime = cache_file.stat().st_mtime
                age = current_time - mtime
                
                if age > min_age_seconds:
                    file_size = cache_file.stat().st_size
                    cache_file.unlink()
                    count += 1
                    total_size += file_size
                    logger.debug(f"Removido cache antigo: {cache_file} (idade: {age / 86400:.1f} dias)")
            except Exception as e:
                logger.error(f"Erro ao processar arquivo de cache {cache_file}: {e}")
        
        logger.info(f"Limpeza de cache: {count} arquivos antigos removidos, {total_size / (1024*1024):.2f} MB liberados")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Obtém estatísticas sobre o uso do cache.
        
        Returns:
            Dict: Estatísticas do cache
        """
        # Estatísticas básicas de hit/miss
        stats = self.stats.copy()
        
        # Adicionar estatísticas de armazenamento
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
        Obtém informações sobre um cache específico ou todos os caches.
        
        Args:
            cache_key: Chave específica do cache ou None para todos
            
        Returns:
            Dict: Informações sobre o cache
        """
        index_file = self.cache_dir / "cache_index.json"
        
        if not index_file.exists():
            return {"error": "Índice de cache não encontrado"}
        
        try:
            with open(index_file, 'r') as f:
                index = json.load(f)
            
            if cache_key:
                return index["entries"].get(cache_key, {"error": "Cache não encontrado"})
            else:
                return index
        except Exception as e:
            return {"error": f"Erro ao ler índice de cache: {e}"}

def cached_function(
    cache_instance: SmartCache, 
    dependencies: Dict[str, Any] = None, 
    cache_name: str = None,
    subfolder: str = None,
    force_recompute: bool = False
):
    """
    Decorator para funções que devem usar o sistema de cache inteligente.
    
    Args:
        cache_instance: Instância do SmartCache
        dependencies: Valores que, quando alterados, invalidam o cache
        cache_name: Nome opcional para o cache
        subfolder: Subpasta opcional dentro do diretório de cache
        force_recompute: Se True, força recomputação
        
    Returns:
        Callable: Função decorada
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
