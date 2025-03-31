"""
Early stopping callback for RF-DETR training
"""

class EarlyStoppingCallback:
    """
    Early stopping callback that monitors mAP and stops training if no improvement 
    over a threshold is observed for a specified number of epochs.
    
    Args:
        patience (int): Number of epochs with no improvement to wait before stopping
        min_delta (float): Minimum change in mAP to qualify as improvement
        use_ema (bool): Whether to use EMA model metrics for early stopping
        verbose (bool): Whether to print early stopping messages
    """
    
    def __init__(self, model, patience=5, min_delta=0.001, use_ema=False, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.use_ema = use_ema
        self.verbose = verbose
        self.best_map = 0.0
        self.counter = 0
        self.model = model
        
    def update(self, log_stats):
        """Update early stopping state based on epoch validation metrics"""
        # Get the mAP value from the log stats
        if self.use_ema and 'ema_test_coco_eval_bbox' in log_stats:
            current_map = log_stats['ema_test_coco_eval_bbox'][0]
        elif 'test_coco_eval_bbox' in log_stats:
            current_map = log_stats['test_coco_eval_bbox'][0]
        else:
            # No valid mAP metric found, skip early stopping check
            return
        
        # Check if current mAP is better than best so far (by at least min_delta)
        if current_map > self.best_map + self.min_delta:
            # We have an improvement
            self.best_map = current_map
            self.counter = 0
            if self.verbose:
                print(f"Early stopping: mAP improved to {current_map:.4f}")
        else:
            # No improvement
            self.counter += 1
            if self.verbose:
                print(f"Early stopping: No improvement in mAP for {self.counter} epochs (best: {self.best_map:.4f}, current: {current_map:.4f})")
            
            # Check if early stopping criteria met
            if self.counter >= self.patience:
                print(f"Early stopping triggered: No improvement above {self.min_delta} threshold for {self.patience} epochs")
                # Request model to stop early
                if self.model:
                    self.model.request_early_stop()