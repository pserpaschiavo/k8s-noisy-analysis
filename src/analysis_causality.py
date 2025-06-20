"""
Module: analysis_causality.py
Description: Causality analysis utilities for multi-tenant time series analysis, including graph visualization.

This module implements methods for causality analysis between time series of different tenants,
including Granger causality and Transfer Entropy (TE), as well as graph visualizations.
"""
import os
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import networkx as nx
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tools.sm_exceptions import MissingDataError
from functools import lru_cache
import warnings

# Import utilities for time series processing
try:
    from src.utils_timeseries import check_and_transform_timeseries, resample_and_align_timeseries
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    logging.warning("utils_timeseries module not available. Some optimizations will not be applied.")

# Configure logging
logger = logging.getLogger(__name__)

# Suppress specific statsmodels warnings
warnings.filterwarnings("ignore", "Maximum Likelihood optimization failed to converge")
warnings.filterwarnings("ignore", "Unable to solve the eigenvalue problem")

# Import the pyinform library for more robust Transfer Entropy
try:
    from pyinform.transferentropy import transfer_entropy
    PYINFORM_AVAILABLE = True
    logger.info("pyinform available. Using optimized implementation for Transfer Entropy.")
except ImportError:
    PYINFORM_AVAILABLE = False
    logger.warning("pyinform is not installed. Transfer Entropy will use a basic implementation. Install with: pip install pyinform")

plt.style.use('tableau-colorblind10')

def plot_causality_graph(causality_matrix: pd.DataFrame, out_path: str, threshold: float = 0.05, directed: bool = True, metric: str = '', metric_color: str = ''):
    """
    Plots a causality graph from a causality matrix (e.g., Granger p-values or scores).
    Edges are drawn where causality_matrix.loc[src, tgt] < threshold.
    Edge width = intensity (1-p or score), color = metric.
    """
    if causality_matrix.empty:
        return None
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    tenants = causality_matrix.index.tolist()
    G.add_nodes_from(tenants)
    edge_labels = {}
    edges = []
    edge_weights = []
    edge_colors = []
    # Color palette for metrics
    metric_palette = {
        'cpu_usage': 'tab:blue',
        'memory_usage': 'tab:orange',
        'disk_io': 'tab:green',
        'network_io': 'tab:red',
    }
    color = metric_color if metric_color else metric_palette.get(metric, 'tab:blue')
    for src in tenants:
        for tgt in tenants:
            if src != tgt:
                val = causality_matrix.at[src, tgt]
                if not pd.isna(val) and float(val) < threshold:
                    weight = 1 - float(val)
                    G.add_edge(src, tgt, weight=weight)
                    edges.append((src, tgt))
                    edge_weights.append(weight * 6 + 1)
                    edge_colors.append(color)
                    edge_labels[(src, tgt)] = f"{weight:.2f}"
    
    # Implement fixed positions for nodes to ensure visual consistency between different graphs
    # First, ensure tenants are in a consistent order for positioning
    sorted_tenants = sorted(tenants)
    
    # Create a dictionary of fixed positions for each tenant (using circular layout)
    pos = {}
    if len(sorted_tenants) <= 1:
        if sorted_tenants:
            pos[sorted_tenants[0]] = np.array([0.0, 0.0])
    else:
        angles = np.linspace(0, 2 * np.pi, len(sorted_tenants), endpoint=False)
        # Position nodes in a circle with radius 0.8 to leave space for labels
        radius = 0.8
        for tenant, angle in zip(sorted_tenants, angles):
            pos[tenant] = np.array([radius * np.cos(angle), radius * np.sin(angle)])
    
    # Add a small jitter to avoid perfect overlap of bidirectional edges
    for node in pos:
        pos[node] = pos[node] + np.random.normal(0, 0.02, size=2)
    
    plt.figure(figsize=(10, 10))  # Increase figure size
    
    # Add a light background for better visibility
    ax = plt.gca()
    ax.set_facecolor('#f8f8f8')
    
    # Draw larger nodes with borders
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=1600, 
                          edgecolors='darkblue', linewidths=1.5)
    
    # Improve node labels
    nx.draw_networkx_labels(G, pos, font_size=13, font_weight='bold', font_color='black')
    
    # Draw each edge individually to apply weight and color
    for idx, (edge, w, c) in enumerate(zip(edges, edge_weights, edge_colors)):
        nx.draw_networkx_edges(G, pos, edgelist=[edge], arrowstyle='->' if directed else '-', 
                             arrows=directed, width=w, edge_color=c, alpha=0.9,
                             connectionstyle='arc3,rad=0.2')  # Curve edges for better visualization
    
    # Improve edge labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, 
                               font_color='darkred', font_size=11, font_weight='bold',
                               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
    
    plt.title(f'Causality Graph ({"Directed" if directed else "Undirected"})\nMetric: {metric if metric else "?"} | Edges: p < {threshold:.2g}',
             fontsize=14, fontweight='bold')
    plt.axis('off')
    # Custom legend
    legend_elements = [
        mlines.Line2D([0], [0], color=color, lw=3, label=f'{metric if metric else "Metric"}')
    ]
    plt.legend(handles=legend_elements, loc='lower left')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    return out_path

def _transfer_entropy(x, y, bins=5, k=1):
    """
    Calculates the Transfer Entropy (TE) from y to x (y→x) using an optimized implementation.
    
    Args:
        x: 1D array representing the target time series
        y: 1D array representing the source time series
        bins: Number of bins for discretization (default=5)
        k: History of the target series to consider (default=1)
        
    Returns:
        Scalar TE value (y→x): how much y helps predict x beyond the history of x
    """
    # Check input data
    min_points = 8  # Reduced from 10 to allow more calculations
    if len(x) != len(y):
        logger.warning(f"Unequal lengths for TE: x={len(x)}, y={len(y)}. Aligning series.")
        # Truncate to the shorter length
        length = min(len(x), len(y))
        x = x[:length]
        y = y[:length]
    
    if len(x) < min_points:
        logger.warning(f"Insufficient data for TE: {len(x)} points available. Minimum {min_points} points.")
        return 0.0
    
    # Handle missing values (NaN)
    x = np.nan_to_num(x, nan=np.nanmean(x))
    y = np.nan_to_num(y, nan=np.nanmean(y))
    
    # If pyinform is available, use the optimized implementation
    if 'transfer_entropy' in globals() and PYINFORM_AVAILABLE:
        try:
            # Use the more robust implementation from the pyinform library
            # Automatic normalization and binning of series
            # Convert to integers as required by pyinform
            x_norm = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)
            y_norm = (y - np.min(y)) / (np.max(y) - np.min(y) + 1e-8)
            
            x_bin = np.floor(x_norm * (bins-1)).astype(int)
            y_bin = np.floor(y_norm * (bins-1)).astype(int)
            
            # Calculate TE using pyinform
            from pyinform.transferentropy import transfer_entropy
            te_value = transfer_entropy(y_bin, x_bin, k=k, local=False)
            return float(te_value)  # Ensure scalar type
        except Exception as e:
            logger.warning(f"Error using pyinform for TE: {e}. Using basic implementation.")
    
    # Basic fallback implementation using numpy histograms
    # Discretize the series
    x_binned = np.digitize(x, np.histogram_bin_edges(x, bins=bins))
    y_binned = np.digitize(y, np.histogram_bin_edges(y, bins=bins))
    
    # Calculate joint and conditional probabilities
    px = np.histogram(x_binned[1:], bins=bins, density=True)[0]
    pxy = np.histogram2d(x_binned[:-1], x_binned[1:], bins=bins, density=True)[0]
    pxyy = np.histogramdd(np.stack([x_binned[:-1], y_binned[:-1], x_binned[1:]], axis=1), 
                         bins=(bins, bins, bins), density=True)[0]
    
    # Add regularization to avoid log(0)
    pxyy = pxyy + 1e-12
    pxy = pxy + 1e-12
    px = px + 1e-12
    
    # TE(y→x) = sum p(x_{t+1}, x_t, y_t) * log [p(x_{t+1}|x_t, y_t) / p(x_{t+1}|x_t)]
    te = 0.0
    for i in range(bins):
        for j in range(bins):
            for k_ in range(bins):
                pxyz = pxyy[j, k_, i]
                pxz = pxy[j, i]
                px_i = px[i]
                if pxyz > 0 and pxz > 0 and px_i > 0:
                    te += pxyz * np.log((pxyz / (np.sum(pxyy[j, k_, :]) + 1e-12)) / 
                                       (pxz / (np.sum(pxy[j, :]) + 1e-12)))
    return te

class CausalityAnalyzer:
    """
    Class responsible for causality calculations (e.g., Granger, Transfer Entropy) between tenants.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def _granger_causality_test(self, source_series: np.ndarray, target_series: np.ndarray, max_lag: int = 3) -> dict:
        """
        Executes a Granger causality test between two time series.
        
        Args:
            source_series: Source time series (potential cause)
            target_series: Target time series (potential effect)
            max_lag: Maximum number of lags to test
            
        Returns:
            Dictionary with test results or an empty dictionary if it fails
        """
        try:
            # Check if the series are constant (very low standard deviation)
            if np.std(source_series) < 1e-6 or np.std(target_series) < 1e-6:
                logger.warning("Constant series detected - cannot run Granger test")
                return {}
                
            # Check for missing values
            if np.isnan(source_series).any() or np.isnan(target_series).any():
                # Try to interpolate missing values
                source_series_pd = pd.Series(source_series).interpolate().bfill().ffill()
                target_series_pd = pd.Series(target_series).interpolate().bfill().ffill()
                
                # Convert back to numpy arrays
                source_series = source_series_pd.to_numpy()
                target_series = target_series_pd.to_numpy()
                
                # Check if NaNs still exist after interpolation
                if np.isnan(source_series).any() or np.isnan(target_series).any():
                    logger.warning("Missing values could not be corrected by interpolation")
                    return {}
            
            # Adjust max_lag if the series is too short
            if len(source_series) <= max_lag + 2:
                new_max_lag = max(1, len(source_series) // 3 - 1)
                logger.warning(f"Series too short for max_lag={max_lag}. Adjusting to {new_max_lag}")
                max_lag = new_max_lag
            
            # Create a DataFrame with the two series
            data = pd.DataFrame({
                'target': target_series,
                'source': source_series
            })
            
            # Execute the Granger test
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                results = grangercausalitytests(data, maxlag=max_lag, verbose=False)
                
            return results
        except Exception as e:
            logger.warning(f"Error in Granger causality test: {str(e)}")
            return {}

    @lru_cache(maxsize=32)
    def compute_granger_matrix(self, metric: str, phase: str, round_id: str, maxlag: int = 5) -> pd.DataFrame:
        """
        Calculates the matrix of p-values from the Granger causality test between all tenants for a specific metric.
        Results are cached to improve performance on repeated runs.
        
        Args:
            metric: Name of the metric for analysis
            phase: Experimental phase (e.g., "1-Baseline", "2-CPU-Noise")
            round_id: ID of the round (e.g., "round-1")
            maxlag: Maximum number of lags for the Granger test
            
        Returns:
            DataFrame where mat[i,j] is the lowest p-value of j causing i (considering lags from 1 to maxlag)
        """
        logger.info(f"Calculating Granger causality matrix for {metric}, {phase}, {round_id}, maxlag={maxlag}")
        subset = self.df[(self.df['metric_name'] == metric) & 
                        (self.df['experimental_phase'] == phase) & 
                        (self.df['round_id'] == round_id)]
        
        tenants = subset['tenant_id'].unique()
        mat = pd.DataFrame(np.nan, index=tenants, columns=tenants)
        
        # Transform to wide format for time series analysis
        wide = subset.pivot_table(index='timestamp', columns='tenant_id', values='metric_value')
        
        # Ensure all tenants are present as columns, filling with NaN if necessary
        wide = wide.reindex(columns=tenants, fill_value=np.nan)

        # Interpolation and filling to handle missing values
        wide = wide.sort_index().interpolate(method='time').ffill().bfill()
        
        # Test Granger causality for each pair of tenants
        for target in tenants:
            for source in tenants:
                if target == source:
                    continue
                
                try:
                    # Select only data with overlapping timestamps
                    data = pd.concat([wide[target], wide[source]], axis=1)
                    data = data.dropna()
                    
                    # Additional check to ensure sufficient data
                    if len(data) < maxlag + 5:  # Requires at least 5 points beyond maxlag
                        logger.warning(f"Insufficient data for Granger test {source}->{target}: {len(data)} points, minimum {maxlag+5}")
                        continue
                        
                    # Check for stationarity before the test
                    from statsmodels.tsa.stattools import adfuller
                    # First-order differentiation if not stationary
                    for col_idx, col_name in enumerate(data.columns):
                        adf_result = adfuller(data[col_name], autolag='AIC')
                        if adf_result[1] > 0.05:  # p-value > 0.05 indicates non-stationarity
                            original_len = len(data)
                            data[col_name] = data[col_name].diff().dropna()
                            logger.warning(f"Series {col_name} was not stationary. Differentiated and reduced from {original_len} to {len(data)} points.")
                            
                    data = data.dropna()  # Remove NaN values after differentiation
                    
                    if len(data) <= maxlag + 3:  # Check again after differentiation
                        logger.warning(f"Insufficient data after differentiation: {len(data)} points")
                        continue
                        
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        # Perform Granger test with improved error handling
                        test_results = grangercausalitytests(
                            data, 
                            maxlag=maxlag, 
                            verbose=False
                        )
                        
                        # Extract the lowest p-value among all lags
                        p_values = [test_results[lag][0]['ssr_chi2test'][1] for lag in range(1, maxlag+1)]
                        min_p_value = min(p_values) if p_values else np.nan
                    
                    # Store the result in the matrix
                    mat.loc[target, source] = min_p_value
                except KeyError:
                    continue
                except MissingDataError:
                    continue
                except Exception as e:
                    logging.warning(f"Error calculating Granger for {source}->{target}: {str(e)}")
                    continue
                    
        return mat

    @lru_cache(maxsize=32)
    def compute_transfer_entropy_matrix(self, metric: str, phase: str, round_id: str, bins: int = 8, k: int = 1) -> pd.DataFrame:
        """
        Calculates the Transfer Entropy (TE) matrix between all tenants for a specific metric.
        Results are cached to improve performance on repeated runs.
        
        Args:
            metric: Name of the metric for analysis
            phase: Experimental phase (e.g., "1-Baseline", "2-CPU-Noise")
            round_id: ID of the round (e.g., "round-1") 
            bins: Number of bins for discretizing continuous series
            k: History of the target series to consider
            
        Returns:
            DataFrame where mat[i,j] is the Transfer Entropy value from j to i (j→i)
            Higher values indicate greater information transfer
        """
        # Log the start of the calculation
        logger.info(f"Calculating Transfer Entropy matrix for {metric} in {phase} ({round_id}), bins={bins}, k={k}")
        
        # Filter relevant data
        subset = self.df[(self.df['metric_name'] == metric) & 
                        (self.df['experimental_phase'] == phase) & 
                        (self.df['round_id'] == round_id)]
        
        tenants = subset['tenant_id'].unique()
        mat = pd.DataFrame(np.nan, index=tenants, columns=tenants)
        
        # Transform to wide format for analysis
        wide = subset.pivot_table(index='timestamp', columns='tenant_id', values='metric_value')

        # Ensure all tenants are present as columns, filling with NaN if necessary
        wide = wide.reindex(columns=tenants, fill_value=np.nan)
        
        # Interpolate and fill NaNs to align all tenants
        wide = wide.sort_index().interpolate(method='time').ffill().bfill()
        
        # Calculate TE for each pair of tenants
        for target in tenants:
            for source in tenants:
                if target == source:
                    continue
                
                try:
                    # Get time series for the tenants
                    target_series = wide[target]
                    source_series = wide[source]
                    
                    # Align indices to ensure temporal correspondence
                    common_idx = target_series.index.intersection(source_series.index)
                    target_values = target_series.loc[common_idx].values
                    source_values = source_series.loc[common_idx].values
                    
                    # Check if there are enough points for a meaningful calculation
                    if len(target_values) >= 8:  # Reduced from 10 to 8 minimum points
                        # Calculate TE and store it in the matrix
                        try:
                            te_value = _transfer_entropy(target_values, source_values, bins=bins, k=k)
                            mat.loc[target, source] = te_value
                        except Exception as calc_error:
                            logging.warning(f"Error in TE calculation for {source}->{target}: {calc_error}")
                    else:
                        logging.warning(f"Insufficient time series for pair {source}->{target}: {len(target_values)} points")
                        
                except Exception as e:
                    logging.error(f"Error calculating Transfer Entropy for {source}->{target}: {str(e)}")
        
        return mat

class CausalityVisualizer:
    """
    Class responsible for causality visualizations (e.g., graphs, heatmaps).
    """
    @staticmethod
    def plot_causality_graph(causality_matrix: pd.DataFrame, out_path: str, threshold: float = 0.05, directed: bool = True, metric: str = '', metric_color: str = ''):
        return plot_causality_graph(causality_matrix, out_path, threshold, directed, metric, metric_color)

    @staticmethod
    def plot_causality_graph_multi(
        causality_matrices: dict,  # {metric: matrix}
        out_path: str,
        threshold: float = 0.05,
        directed: bool = True,
        metric_palette: dict = {},
        threshold_mode: str = ''  # 'p' for Granger, 'TE' for Transfer Entropy
    ):
        """
        Plots a causality graph comparing multiple metrics.
        Each metric is a different edge color.
        threshold_mode: 'p' (edges p < threshold), 'TE' (edges TE > threshold), or '' (auto).
        """
        if not causality_matrices:
            return None
        # Default palette
        if not metric_palette:
            metric_palette = {
                'cpu_usage': 'tab:blue',
                'memory_usage': 'tab:orange',
                'disk_io': 'tab:green',
                'network_io': 'tab:red',
            }
        # --- Enhanced logic for threshold_mode and legend ---
        # Detect for each metric if it is p-value (real Granger) or TE
        metric_modes = {}
        for metric, mat in causality_matrices.items():
            if mat.isnull().all().all():
                metric_modes[metric] = 'unknown'
            elif (mat.max().max() <= 1.0) and (mat.min().min() >= 0.0):
                # Could be p-value (real or placeholder Granger)
                # If not a placeholder (all NaN), consider it a p-value
                metric_modes[metric] = 'p'
            else:
                metric_modes[metric] = 'TE'
        # Prioritize p-value if there is at least one real Granger metric
        if threshold_mode == '':
            if 'p' in metric_modes.values():
                threshold_mode = 'p'
            else:
                threshold_mode = 'TE'
        # Unite all nodes
        all_tenants = set()
        for mat in causality_matrices.values():
            all_tenants.update(mat.index.tolist())
        tenants = sorted(all_tenants)
        G = nx.DiGraph() if directed else nx.Graph()
        G.add_nodes_from(tenants)
        edge_labels = {}
        edge_colors = []
        edge_widths = []
        edge_list = []
        edge_metrics = []
        for metric, mat in causality_matrices.items():
            color = metric_palette.get(metric, 'tab:blue')
            mode = metric_modes.get(metric, threshold_mode)
            for src in tenants:
                for tgt in tenants:
                    if src != tgt and src in mat.index and tgt in mat.columns:
                        val = mat.at[src, tgt]
                        if mode == 'p':
                            cond = (not pd.isna(val)) and (float(val) < threshold)
                        else:
                            cond = (not pd.isna(val)) and (float(val) > threshold)
                        if cond:
                            weight = 1 - float(val) if mode == 'p' else float(val)
                            G.add_edge(src, tgt)
                            edge_list.append((src, tgt))
                            edge_colors.append(color)
                            edge_widths.append(weight * 6 + 1)
                            edge_labels[(src, tgt)] = f"{weight:.2f}"
                            edge_metrics.append(metric)
        # Circular layout to ensure visibility
        pos = nx.circular_layout(G)
        plt.figure(figsize=(8, 8))
        nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=1200)
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
        # Draw each edge individually, with offset and visible arrow
        for (edge, color, width, metric) in zip(edge_list, edge_colors, edge_widths, edge_metrics):
            nx.draw_networkx_edges(
                G, pos, edgelist=[edge],
                arrowstyle='-|>' if directed else '-',
                arrows=directed,
                width=width,
                edge_color=color,
                alpha=0.8,
                connectionstyle='arc3,rad=0.18',
                min_source_margin=25, min_target_margin=25,
                arrowsize=28
            )
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='gray', font_size=10)
        # Legend for each metric
        legend_elements = [mlines.Line2D([0], [0], color=metric_palette.get(m, 'tab:blue'), lw=3, label=m) for m in causality_matrices.keys()]
        plt.legend(handles=legend_elements, loc='lower left')
        # Automatic contextual legend
        if threshold_mode == 'p':
            legend_str = f'Edges: p-value < {threshold:.2g}'
        else:
            legend_str = f'Edges: TE > {threshold:.2g}'
        plt.title(f'Causality Graph (Metric Comparison) | {legend_str}')
        plt.axis('off')
        plt.tight_layout()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path)
        plt.close()
        return out_path

class CausalityModule:
    """
    High-level module for causality analysis, integrating calculation and visualization.
    """
    def __init__(self, df: pd.DataFrame):
        self.analyzer = CausalityAnalyzer(df)
        self.visualizer = CausalityVisualizer()

    def run_granger_and_plot(self, metric: str, phase: str, round_id: str, out_dir: str, maxlag: int = 5, threshold: float = 0.05):
        os.makedirs(out_dir, exist_ok=True)
        mat = self.analyzer.compute_granger_matrix(metric, phase, round_id, maxlag)
        out_path = os.path.join(out_dir, f"causality_graph_granger_{metric}_{phase}_{round_id}.png")
        # Color palette same as used in the plot function
        metric_palette = {
            'cpu_usage': 'tab:blue',
            'memory_usage': 'tab:orange',
            'disk_io': 'tab:green',
            'network_io': 'tab:red',
        }
        color = metric_palette.get(metric, 'tab:blue')
        return self.visualizer.plot_causality_graph(mat, out_path, threshold, directed=True, metric=metric, metric_color=color)

    def run_transfer_entropy_and_plot(self, metric: str, phase: str, round_id: str, out_dir: str, bins: int = 8, threshold: float = 0.05):
        os.makedirs(out_dir, exist_ok=True)
        mat = self.analyzer.compute_transfer_entropy_matrix(metric, phase, round_id, bins)
        out_path = os.path.join(out_dir, f"causality_graph_te_{metric}_{phase}_{round_id}.png")
        # Color palette same as used in the plot function
        metric_palette = {
            'cpu_usage': 'tab:blue',
            'memory_usage': 'tab:orange',
            'disk_io': 'tab:green',
            'network_io': 'tab:red',
        }
        color = metric_palette.get(metric, 'tab:blue')
        # For TE, threshold highlights stronger relationships (TE > threshold)
        def plot_te_graph(te_matrix, out_path, threshold, metric, color):
            if te_matrix.empty:
                return None
            G = nx.DiGraph()
            tenants = te_matrix.index.tolist()
            G.add_nodes_from(tenants)
            edge_labels = {}
            edges = []
            edge_weights = []
            edge_colors = []
            for src in tenants:
                for tgt in tenants:
                    if src != tgt:
                        val = te_matrix.at[src, tgt]
                        if not pd.isna(val) and float(val) > threshold:
                            weight = float(val)
                            G.add_edge(src, tgt, weight=weight)
                            edges.append((src, tgt))
                            edge_weights.append(weight * 6 + 1)
                            edge_colors.append(color)
                            edge_labels[(src, tgt)] = f"{weight:.2f}"
            pos = nx.spring_layout(G, seed=42)
            plt.figure(figsize=(7, 7))
            nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=1200)
            nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
            for idx, (edge, w, c) in enumerate(zip(edges, edge_weights, edge_colors)):
                nx.draw_networkx_edges(G, pos, edgelist=[edge], arrowstyle='->', arrows=True, width=w, edge_color=c, alpha=0.8)
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='gray', font_size=10)
            plt.title(f'Transfer Entropy Graph (TE > {threshold:.2g})\nMetric: {metric}')
            plt.axis('off')
            legend_elements = [
                mlines.Line2D([0], [0], color=color, lw=3, label=f'{metric}')
            ]
            plt.legend(handles=legend_elements, loc='lower left')
            plt.tight_layout()
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            plt.savefig(out_path)
            plt.close()
            return out_path
        return plot_te_graph(mat, out_path, threshold, metric, color)
