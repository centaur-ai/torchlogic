import numpy as np

## Placeholder for the Maximial Information Coefficient (MIC) estimation implementation from Minepy, which is not available in python >=3.11

# A minimal approximation of Maximial Information Coeff estimation (Reshef, 2011) https://pmc.ncbi.nlm.nih.gov/articles/PMC3325791/
# Only dependent on numpy. Slow as dirt in testing (~20x slower) with about 20% MAPE in comparison to the canonical package estimation time and estimates.

## Pseudo-code:
# https://pmc.ncbi.nlm.nih.gov/articles/instance/3325791/bin/NIHMS358982-supplement-Supplemental_Figures_and_Tables.pdf


#### WISH LIST ####
# - Adaptive binning based on data distribution (e.g., quantiles) (Can also try equi-depth binning)
# - More efficient mutual information calculation (currently uses nested loops)
# - Math review for correctness, especially in normalization, binning, and parameter choices
# - More verbose error handling and logging

def bin_series(x, bins):
    """Assigns each value to a bin index."""
    # Convert input to numpy array and flatten to 1D
    x = np.asarray(x).flatten()
    
    # Handle edge case of empty array
    if len(x) == 0:
        return np.array([], dtype=int)
    
    # Handle edge case where all values are identical (zero variance)
    if np.var(x) == 0:
        return np.zeros(len(x), dtype=int)
    
    # Check if requested bins is less than or equal to unique values
    if bins <= len(np.unique(x)):
        # Use numpy's histogram to create bin edges
        _, edges = np.histogram(x, bins=bins)
        # Digitize assigns each value to a bin, subtract 1 for 0-based indexing
        binned = np.digitize(x, edges[:-1], right=False) - 1
        # Clip values to ensure they're within valid bin range
        binned = np.clip(binned, 0, bins - 1)
        return binned
    else:
        # If more bins requested than unique values, use unique value mapping
        unique_vals = np.unique(x)
        return np.searchsorted(unique_vals, x)


def mutual_information_2d(x_binned, y_binned):

    try:
        # Ensure inputs are integer arrays for indexing
        x_binned = np.asarray(x_binned, dtype=int)
        y_binned = np.asarray(y_binned, dtype=int)
        
        #  Determine the size of the joint histogram matrix
        max_x = int(np.max(x_binned)) + 1
        max_y = int(np.max(y_binned)) + 1
        
        # Initialize joint histogram matrix with zeros
        joint_hist = np.zeros((max_x, max_y))
        
        # Populate joint histogram by counting co-occurrences
        for i in range(len(x_binned)):
            joint_hist[x_binned[i], y_binned[i]] += 1
        
        # Convert counts to probabilities
        n = len(x_binned)
        joint_prob = joint_hist / n
        
        # Calculate marginal probabilities
        px = np.sum(joint_prob, axis=1)  # Sum over y-axis to get P(X)
        py = np.sum(joint_prob, axis=0)  # Sum over x-axis to get P(Y)
        
        ####### These nested loops are very inefficient. Could vectorize this calculation? ########
        # Calculate mutual information using the formula: MI = sum(P(x,y) * log(P(x,y) / (P(x) * P(y))))
        mi = 0.0
        for i in range(max_x):
            for j in range(max_y):
                if joint_prob[i, j] > 0 and px[i] > 0 and py[j] > 0:
                    mi += joint_prob[i, j] * np.log(joint_prob[i, j] / (px[i] * py[j]))
        
        # Ensure MI is non-negative (theoretically) it should be, but you never know for sure)
        return max(mi, 0.0)
    except Exception as e:
        # Generic exception handling - prints error and returns 0
        print(f"MI calculation error: {e}")
        return 0.0
    
def compute_mic(x, y, alpha=0.6, c=5, max_bins=None):
    """Compute the Maximial Information Coefficient (MIC) between two variables x and y"""
    
    # Convert inputs to flattened numpy arrays
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    # Handle edge case of (x,y) length mismatch
    if len(x) != len(y):
        raise ValueError(f"x and y must have same length: {len(x)} vs {len(y)}")
    
    # Handle edge case of very small datasets
    if len(x) < 4:
        return 0.0, (0, 0)
    
    # Calculate sample size
    n = len(x)

    #### Alpha scaling logic seems questionable ####
    # Calculate Maximum number of bins based on sample size and alpha
    B = min(int(n ** alpha), 500) if alpha <= 1 else min(int(n ** (alpha / 100.0)), 500)
    max_bins = min(int(np.sqrt(B)), 25) if max_bins is None else max_bins
    
    max_mic = 0.0
    best_grid = (0, 0)

    # Iterate through all possible grid sizes
    for x_bins in range(2, max_bins + 1):
        for y_bins in range(2, max_bins + 1):
            
            # Skip if grid size exceeds B parameter
            if x_bins * y_bins >= B:
                continue

            # This is a heuristic to avoid extreme binning ratios
            if max(x_bins, y_bins) > c * min(x_bins, y_bins):
                continue
            
            # Skip if binning resulted in fewer than 2 unique bins
            try:
                # Bin the data using the current grid size
                x_binned = bin_series(x, x_bins)
                y_binned = bin_series(y, y_bins)
                if len(np.unique(x_binned)) < 2 or len(np.unique(y_binned)) < 2:
                    continue
                
                # Calculate MIC score and update best if it's better
                mi = mutual_information_2d(x_binned, y_binned)
                
                # Normalize MI by log of minimum grid dimension
           ###### I am not very confident this normalization is corrrect ######
                norm = np.log(min(x_bins, y_bins))
                if norm > 0 and mi > 0:
                    mic = mi / norm
                    if mic > max_mic:
                        max_mic = mic
                        best_grid = (x_bins, y_bins)
            
            # Skip this grid size if any error occurs
            except Exception:
                continue
    
    # Return the best MIC score and corresponding grid size
    return max_mic, best_grid


## I asked claude to wrap everything up in a single function so we can run all cov at once, and output the desired matrix
## Implementation is a bit verbose, but it handles both single covariate MIC and pairwise (X) MIC matrix computation.
def compute_mic_matrix(X, y=None, alpha=0.6, c=5, max_bins=None):
    """
    Compute MIC for multiple covariates.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features) or (n_samples,)
        Input covariates. If 2D, each column is a covariate.
    y : array-like, shape (n_samples,), optional
        Target variable. If None, computes pairwise MIC between all covariates in X.
    alpha, c, max_bins : parameters passed to compute_mic_e
    
    Returns:
    --------
    If y is provided:
        - mic_scores : array, shape (n_features,)
        - best_grids : list of tuples, shape (n_features,)
    
    If y is None:
        - mic_matrix : array, shape (n_features, n_features)
        - best_grids : list of lists of tuples
    """
    X = np.asarray(X)
    
    # Handle 1D input
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    n_samples, n_features = X.shape
    
    # Not really needed, but claude included handling for no-target case.
    if y is not None:
        # Compute MIC between each covariate and target y
        y = np.asarray(y).flatten()
        if len(y) != n_samples:
            raise ValueError(f"X and y must have same number of samples: {n_samples} vs {len(y)}")
        
        mic_scores = np.zeros(n_features)
        best_grids = []
        
        for i in range(n_features):
            mic_score, best_grid = compute_mic(X[:, i], y, alpha, c, max_bins)
            mic_scores[i] = mic_score
            best_grids.append(best_grid)
        
        return mic_scores, best_grids
    
    else:
        # Compute pairwise MIC matrix between all covariates
        mic_matrix = np.zeros((n_features, n_features))
        best_grids = [[None for _ in range(n_features)] for _ in range(n_features)]
        
        for i in range(n_features):
            for j in range(i, n_features):
                if i == j:
                    mic_matrix[i, j] = 1.0  # Perfect correlation with itself
                    best_grids[i][j] = (0, 0)
                else:
                    mic_score, best_grid = compute_mic(X[:, i], X[:, j], alpha, c, max_bins)
                    mic_matrix[i, j] = mic_score
                    mic_matrix[j, i] = mic_score  # Symmetric
                    best_grids[i][j] = best_grid
                    best_grids[j][i] = best_grid
        
        return mic_matrix, best_grids