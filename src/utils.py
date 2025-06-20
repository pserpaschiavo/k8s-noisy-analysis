#!/usr/bin/env python3
"""
Module: utils.py
Description: General utilities for the analysis pipeline.
"""

import warnings
import contextlib
import logging

logger = logging.getLogger(__name__)

@contextlib.contextmanager
def suppress_statsmodels_warnings():
    """
    Context manager to suppress repetitive statsmodels warnings.
    Filters specific warnings while keeping other important ones.
    """
    with warnings.catch_warnings():
        # Filter common statsmodels warnings
        warnings.filterwarnings('ignore', 'Non-stationary starting autoregressive parameters')
        warnings.filterwarnings('ignore', 'Value in x_0 detected')
        warnings.filterwarnings('ignore', message='.*flat prior.*')
        warnings.filterwarnings('ignore', message='.*distribution is not normalized.*')
        warnings.filterwarnings('ignore', message='.*divide by zero.*')
        warnings.filterwarnings('ignore', message='.*invalid value.*')
        
        # Suppress convergence warnings
        warnings.filterwarnings('ignore', message='.*Maximum Likelihood optimization failed.*')
        warnings.filterwarnings('ignore', message='.*The iteration limit.*')
        
        # Other common statistical warnings
        warnings.filterwarnings('ignore', message='.*p-value.*')
        
        yield
