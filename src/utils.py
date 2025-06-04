#!/usr/bin/env python3
"""
Module: utils.py
Description: Utilidades gerais para o pipeline de análise.
"""

import warnings
import contextlib
import logging

logger = logging.getLogger(__name__)

@contextlib.contextmanager
def suppress_statsmodels_warnings():
    """
    Context manager para suprimir avisos repetitivos do statsmodels.
    Filtra avisos específicos enquanto mantém outros importantes.
    """
    with warnings.catch_warnings():
        # Filtrar avisos comuns do statsmodels
        warnings.filterwarnings('ignore', 'Non-stationary starting autoregressive parameters')
        warnings.filterwarnings('ignore', 'Value in x_0 detected')
        warnings.filterwarnings('ignore', message='.*flat prior.*')
        warnings.filterwarnings('ignore', message='.*distribution is not normalized.*')
        warnings.filterwarnings('ignore', message='.*divide by zero.*')
        warnings.filterwarnings('ignore', message='.*invalid value.*')
        
        # Suprimir avisos de convergência
        warnings.filterwarnings('ignore', message='.*Maximum Likelihood optimization failed.*')
        warnings.filterwarnings('ignore', message='.*The iteration limit.*')
        
        # Outros avisos estatísticos comuns
        warnings.filterwarnings('ignore', message='.*p-value.*')
        
        yield
