import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    if not seqs:
        return np.empty((0, 0), dtype=int)
    
    # Determine maximum length if max_len is not specified
    if max_len is None:
        max_len = max((len(seq) for seq in seqs), default=0)
        
    N = len(seqs)
    
    # Initialize the array filled with pad_value
    arr = np.full((N, max_len), pad_value, dtype=int)
    
    # Populate the array, handling both padding and truncation automatically
    for i, seq in enumerate(seqs):
        length = min(len(seq), max_len)
        if length > 0:
            arr[i, :length] = seq[:length]
            
    return arr