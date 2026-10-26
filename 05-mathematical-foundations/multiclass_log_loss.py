"""
Multi-Class Log Loss: Derivation, Safe Implementation & Numerical Clipping
Team: Pocket Data Lab
Article Reference: https://malcolmlow.com/2026/10/26/multiclass-log-loss-explained-cross-entropy-mle-python/
"""

import numpy as np

def safe_multiclass_log_loss(y_true, y_pred, eps=1e-15):
    """
    Computes numerically stable multi-class log loss with clipping bounds.
    
    Parameters:
    -----------
    y_true : np.ndarray of shape (N, K)
        One-hot encoded binary ground truth indicators.
    y_pred : np.ndarray of shape (N, K)
        Predicted class probability distribution.
    eps : float, default=1e-15
        Clipping threshold preventing log(0) explosion.
        
    Returns:
    --------
    float: Mean multi-class log loss.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    # 1. Clip probabilities to [eps, 1 - eps]
    y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)
    
    # 2. Re-normalize across classes
    y_pred_norm = y_pred_clipped / np.sum(y_pred_clipped, axis=1, keepdims=True)
    
    # 3. Compute cross-entropy loss: - (1 / N) * sum(y * ln(p))
    loss = - np.sum(y_true * np.log(y_pred_norm)) / y_true.shape[0]
    return float(loss)

if __name__ == '__main__':
    # Test ground truth: 3 samples, 3 classes
    y_true = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    
    # Well-calibrated prediction
    y_pred_good = np.array([
        [0.85, 0.10, 0.05],
        [0.05, 0.90, 0.05],
        [0.10, 0.10, 0.80]
    ])
    
    # Uniform prior prediction
    y_pred_uniform = np.full_like(y_true, 1.0 / 3.0)
    
    print(f"Good Predictions Loss:    {safe_multiclass_log_loss(y_true, y_pred_good):.5f}")
    print(f"Uniform Prior Loss:       {safe_multiclass_log_loss(y_true, y_pred_uniform):.5f}")
