from collections import Counter
import numpy as np

def mean_median_mode(x: list) -> dict:
    """
    Returns a dictionary with mean, median, and mode.
    """
    arr = np.array(x)

    mean = np.mean(arr)
    median = np.median(arr)

    counts = Counter(arr)
    max_count = max(counts.values())
    mode = min(k for k, v in counts.items() if v == max_count)

    return {
        "mean": float(mean),
        "median": float(median),
        "mode": float(mode)
    }
    
    