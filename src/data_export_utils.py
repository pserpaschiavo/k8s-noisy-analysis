"""
Module: data_export_utils.py
Description: Enhanced utilities for exporting data in various formats
"""
import os
import pandas as pd
import logging
import json
from typing import Dict, Any, Optional, List, Union
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from pathlib import Path

# Setup logging
logger = logging.getLogger(__name__)

class AnalysisDataExporter:
    """
    A class that manages data files for various analysis results in multiple formats.
    Provides utilities for saving, loading, and managing analysis results in Parquet,
    CSV, Excel, JSON, and HTML formats.
    """
    
    def __init__(self, base_output_dir: str = None, default_format: str = "parquet"):
        """
        Initialize the AnalysisDataExporter.
        
        Args:
            base_output_dir: Base directory for output files. If None, uses './data/analysis_outputs'
            default_format: Default output format (parquet, csv, excel, json, html)
        """
        self.base_output_dir = base_output_dir or "./data/analysis_outputs"
        self.default_format = default_format
        os.makedirs(self.base_output_dir, exist_ok=True)
        logger.info(f"AnalysisDataExporter initialized with base directory: {self.base_output_dir}")
    
    def ensure_path_exists(self, filepath: str) -> None:
        """
        Ensure that the directory path for a file exists.
        
        Args:
            filepath: Path to the file
        """
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
    
    def save_analysis_result(self, 
                           df: pd.DataFrame, 
                           analysis_type: str, 
                           round_id: Optional[str] = None,
                           sub_type: Optional[str] = None,
                           custom_name: Optional[str] = None,
                           output_format: Optional[str] = None) -> str:
        """
        Save analysis results to a file with organized directory structure.
        
        Args:
            df: DataFrame with analysis results
            analysis_type: Type of analysis (e.g., 'descriptive', 'impact', 'correlation')
            round_id: Optional round identifier for round-specific results
            sub_type: Optional sub-type of analysis (e.g., 'heatmap', 'time_series')
            custom_name: Optional custom name for the file
            output_format: Output format (parquet, csv, excel, json, html)
            
        Returns:
            Path to the saved file
        """
        if df is None or df.empty:
            logger.warning(f"Cannot save empty DataFrame for {analysis_type} analysis.")
            return ""
        
        # Use default format if none specified
        output_format = output_format or self.default_format
            
        # Create directory structure
        analysis_dir = os.path.join(self.base_output_dir, analysis_type)
        
        if round_id:
            analysis_dir = os.path.join(analysis_dir, round_id)
        
        if sub_type:
            analysis_dir = os.path.join(analysis_dir, sub_type)
            
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Create base filename
        if custom_name:
            base_filename = custom_name
        else:
            if round_id and sub_type:
                base_filename = f"{analysis_type}_{round_id}_{sub_type}"
            elif round_id:
                base_filename = f"{analysis_type}_{round_id}"
            elif sub_type:
                base_filename = f"{analysis_type}_{sub_type}"
            else:
                base_filename = f"{analysis_type}"
        
        # Add appropriate extension based on format
        extensions = {
            'parquet': '.parquet',
            'csv': '.csv',
            'excel': '.xlsx',
            'json': '.json',
            'html': '.html'
        }
        
        filename = base_filename + extensions.get(output_format, '.parquet')
        filepath = os.path.join(analysis_dir, filename)
        
        try:
            # Handle types not supported by output format
            if output_format == 'parquet':
                df = self._handle_unsupported_types(df)
            
            # Ensure path exists
            self.ensure_path_exists(filepath)
            
            # Save DataFrame in the specified format
            self._save_df_in_format(df, filepath, output_format)
            
            logger.info(f"Successfully saved {analysis_type} analysis results to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving {analysis_type} analysis to {filepath}: {e}")
            return ""
    
    def _save_df_in_format(self, df: pd.DataFrame, filepath: str, output_format: str) -> None:
        """
        Save DataFrame in the specified format.
        
        Args:
            df: DataFrame to save
            filepath: Path where to save the file
            output_format: Format to use (parquet, csv, excel, json, html)
        """
        try:
            # Certifique-se de que o DataFrame é válido
            if df is None or not isinstance(df, pd.DataFrame):
                logger.error(f"Invalid DataFrame object: {type(df)}")
                raise ValueError(f"Expected pandas DataFrame, got {type(df)}")
                
            if output_format == 'parquet':
                # Tratar tipos especiais antes de salvar em Parquet
                df = self._handle_unsupported_types(df)
                
                # Usar PyArrow diretamente para mais controle
                try:
                    table = pa.Table.from_pandas(df)
                    pq.write_table(table, filepath)
                except pa.ArrowInvalid as e:
                    logger.error(f"PyArrow error: {e}")
                    # Fallback para CSV se Parquet falhar
                    logger.warning(f"Falling back to CSV format for {filepath}")
                    csv_path = filepath.replace('.parquet', '.csv')
                    df.to_csv(csv_path, index=False)
                    logger.info(f"Saved as CSV instead at {csv_path}")
                    raise
            elif output_format == 'csv':
                df.to_csv(filepath, index=False)
            elif output_format == 'excel':
                df.to_excel(filepath, index=False, engine='openpyxl')
            elif output_format == 'json':
                # For better JSON formatting with complex data
                with open(filepath, 'w', encoding='utf-8') as f:
                    json_str = df.to_json(orient='records', date_format='iso')
                    json_obj = json.loads(json_str)
                    json.dump(json_obj, f, ensure_ascii=False, indent=4)
            elif output_format == 'html':
                html_content = df.to_html(index=False)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(f"""
                    <html>
                    <head>
                        <style>
                            table {{ border-collapse: collapse; width: 100%; }}
                            th, td {{ text-align: left; padding: 8px; border: 1px solid #ddd; }}
                            tr:nth-child(even) {{ background-color: #f2f2f2; }}
                            th {{ background-color: #4CAF50; color: white; }}
                        </style>
                    </head>
                    <body>
                        <h2>{os.path.basename(filepath).replace('.html', '')} Analysis Results</h2>
                        {html_content}
                    </body>
                    </html>
                    """)
            else:
                logger.error(f"Unsupported output format: {output_format}")
        except Exception as e:
            logger.error(f"Error saving DataFrame to {filepath} in {output_format} format: {e}")
            raise
    
    def _handle_unsupported_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert columns with unsupported types to formats compatible with Parquet.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with converted types
        """
        df_copy = df.copy()
        
        # Primeiro, verificar e lidar com tipos incompatíveis com Parquet
        for col in df_copy.columns:
            # Examinar amostra dos dados para determinar o tratamento necessário
            sample = df_copy[col].dropna().head(100)
            
            # Tratar valores NaN/None primeiro
            df_copy[col] = df_copy[col].fillna(pd.NA)
            
            # Converter colunas com tipos complexos
            if df_copy[col].dtype == 'object':
                # Lista dos tipos encontrados nesta coluna
                types_found = set()
                for x in sample:
                    if x is not None and not pd.isna(x):
                        types_found.add(type(x))
                
                # Tratar diferentes tipos de dados
                if types_found:
                    logger.debug(f"Column {col} contains types: {types_found}")
                    
                    # Se contém tipos complexos (dict, set, list, etc)
                    if any(issubclass(t, (dict, set)) for t in types_found):
                        logger.info(f"Converting column {col} with dict/set to JSON strings")
                        df_copy[col] = df_copy[col].apply(
                            lambda x: json.dumps(x) if x is not None and not pd.isna(x) else None
                        )
                    
                    # Se contém arrays numpy
                    elif any(issubclass(t, np.ndarray) for t in types_found):
                        logger.info(f"Converting column {col} with numpy arrays to JSON strings")
                        df_copy[col] = df_copy[col].apply(
                            lambda x: json.dumps(x.tolist()) if isinstance(x, np.ndarray) and x is not None else 
                                     (json.dumps(float(x)) if isinstance(x, (np.generic, np.number)) and x is not None else x)
                        )
                    
                    # Se contém listas ou tuplas
                    elif any(issubclass(t, (list, tuple)) for t in types_found):
                        # Para Parquet, é melhor sempre converter listas para strings JSON
                        # porque mesmo listas simples podem causar problemas em certos casos
                        logger.info(f"Converting column {col} with lists/tuples to JSON strings")
                        df_copy[col] = df_copy[col].apply(
                            lambda x: json.dumps(x) if x is not None and not pd.isna(x) else None
                        )
                    
                    # Objetos personalizados ou tipos não reconhecidos
                    elif not any(issubclass(t, (int, float, str, bool, np.number, np.bool_)) for t in types_found):
                        logger.info(f"Converting column {col} with custom objects to strings")
                        df_copy[col] = df_copy[col].apply(
                            lambda x: str(x) if x is not None and not pd.isna(x) else None
                        )
            
            # Garantir que valores NaN sejam consistentes
            df_copy[col] = df_copy[col].replace({pd.NA: None})
        
        return df_copy
    
    def load_analysis_result(self, 
                           analysis_type: str, 
                           round_id: Optional[str] = None,
                           sub_type: Optional[str] = None,
                           custom_name: Optional[str] = None,
                           input_format: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Load analysis results from a file.
        
        Args:
            analysis_type: Type of analysis (e.g., 'descriptive', 'impact', 'correlation')
            round_id: Optional round identifier for round-specific results
            sub_type: Optional sub-type of analysis (e.g., 'heatmap', 'time_series')
            custom_name: Optional custom name for the file
            input_format: Format of the input file (parquet, csv, excel, json)
            
        Returns:
            DataFrame with analysis results or None if file not found
        """
        # Use default format if none specified
        input_format = input_format or self.default_format
        
        # Create directory structure
        analysis_dir = os.path.join(self.base_output_dir, analysis_type)
        
        if round_id:
            analysis_dir = os.path.join(analysis_dir, round_id)
        
        if sub_type:
            analysis_dir = os.path.join(analysis_dir, sub_type)
            
        # Create base filename
        if custom_name:
            base_filename = custom_name
        else:
            if round_id and sub_type:
                base_filename = f"{analysis_type}_{round_id}_{sub_type}"
            elif round_id:
                base_filename = f"{analysis_type}_{round_id}"
            elif sub_type:
                base_filename = f"{analysis_type}_{sub_type}"
            else:
                base_filename = f"{analysis_type}"
        
        # Add appropriate extension based on format
        extensions = {
            'parquet': '.parquet',
            'csv': '.csv',
            'excel': '.xlsx',
            'json': '.json',
            'html': '.html'
        }
        
        filename = base_filename + extensions.get(input_format, '.parquet')
        filepath = os.path.join(analysis_dir, filename)
        
        try:
            if not os.path.exists(filepath):
                logger.warning(f"Analysis result file not found at {filepath}")
                return None
                
            # Load DataFrame based on format
            if input_format == 'parquet':
                df = pd.read_parquet(filepath)
            elif input_format == 'csv':
                df = pd.read_csv(filepath)
            elif input_format == 'excel':
                df = pd.read_excel(filepath, engine='openpyxl')
            elif input_format == 'json':
                df = pd.read_json(filepath, orient='records')
            else:
                logger.error(f"Unsupported input format: {input_format}")
                return None
                
            logger.info(f"Successfully loaded {analysis_type} analysis results from {filepath}")
            return df
        except Exception as e:
            logger.error(f"Error loading {analysis_type} analysis from {filepath}: {e}")
            return None
    
    def save_consolidated_results(self, 
                                dfs: Dict[str, pd.DataFrame], 
                                output_path: Optional[str] = None,
                                output_format: Optional[str] = None) -> str:
        """
        Save multiple DataFrames as separate files or as partitions.
        
        Args:
            dfs: Dictionary of DataFrames where keys are partition names
            output_path: Optional custom output path
            output_format: Output format (parquet, csv, excel, json, html)
            
        Returns:
            Path to the saved dataset directory
        """
        if not dfs:
            logger.warning("No DataFrames provided for consolidated saving.")
            return ""
            
        # Use default format if none specified
        output_format = output_format or self.default_format
            
        # Create output path
        output_path = output_path or os.path.join(self.base_output_dir, "consolidated")
        os.makedirs(output_path, exist_ok=True)
        
        try:
            # Save each DataFrame separately
            for name, df in dfs.items():
                if df is None or df.empty:
                    logger.warning(f"Skipping empty DataFrame for {name}")
                    continue
                
                # Extensions based on format
                extensions = {
                    'parquet': '.parquet',
                    'csv': '.csv',
                    'excel': '.xlsx',
                    'json': '.json',
                    'html': '.html'
                }
                
                ext = extensions.get(output_format, '.parquet')
                file_path = os.path.join(output_path, f"{name}{ext}")
                
                # Handle types for Parquet
                if output_format == 'parquet':
                    df = self._handle_unsupported_types(df)
                
                # Save DataFrame in the specified format
                self.ensure_path_exists(file_path)
                self._save_df_in_format(df, file_path, output_format)
                
                logger.info(f"Saved {name} to {file_path}")
                
            logger.info(f"Successfully saved consolidated results to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error saving consolidated results to {output_path}: {e}")
            return ""
    
    def list_available_analyses(self) -> Dict[str, Dict[str, List[str]]]:
        """
        List all available analysis results in the base directory, grouped by format.
        
        Returns:
            Dictionary mapping analysis types to formats to lists of available files
        """
        results = {}
        
        try:
            # Check if base directory exists
            if not os.path.exists(self.base_output_dir):
                logger.warning(f"Base directory {self.base_output_dir} does not exist.")
                return {}
                
            # Iterate through subdirectories (analysis types)
            for analysis_type in os.listdir(self.base_output_dir):
                analysis_path = os.path.join(self.base_output_dir, analysis_type)
                
                if os.path.isdir(analysis_path):
                    # Find all data files for this analysis type
                    format_files = {
                        'parquet': [],
                        'csv': [],
                        'excel': [],
                        'json': [],
                        'html': []
                    }
                    
                    for root, _, files in os.walk(analysis_path):
                        for file in files:
                            rel_path = os.path.relpath(
                                os.path.join(root, file), 
                                analysis_path
                            )
                            
                            if file.endswith('.parquet'):
                                format_files['parquet'].append(rel_path)
                            elif file.endswith('.csv'):
                                format_files['csv'].append(rel_path)
                            elif file.endswith('.xlsx'):
                                format_files['excel'].append(rel_path)
                            elif file.endswith('.json'):
                                format_files['json'].append(rel_path)
                            elif file.endswith('.html'):
                                format_files['html'].append(rel_path)
                    
                    # Only include formats that have files
                    analysis_formats = {fmt: files for fmt, files in format_files.items() if files}
                    if analysis_formats:
                        results[analysis_type] = analysis_formats
            
            return results
        except Exception as e:
            logger.error(f"Error listing available analyses: {e}")
            return {}
