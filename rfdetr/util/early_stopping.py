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
    
    def __init__(self, patience=5, min_delta=0.001, use_ema=False, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.use_ema = use_ema
        self.verbose = verbose
        self.best_map = 0.0
        self.counter = 0
        self.stop_training = False
        self.model = None
        
    def update(self, log_stats):
        """Update early stopping state based on epoch validation metrics"""
        if self.use_ema and 'ema_test_coco_eval_bbox' in log_stats:
            current_map = log_stats['ema_test_coco_eval_bbox'][0]
        elif 'test_coco_eval_bbox' in log_stats:
            current_map = log_stats['test_coco_eval_bbox'][0]
        else:
            return
        
        if current_map > self.best_map + self.min_delta:
            self.best_map = current_map
            self.counter = 0
            if self.verbose:
                print(f"Early stopping: mAP improved to {current_map:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"Early stopping: No improvement in mAP for {self.counter} epochs (best: {self.best_map:.4f}, current: {current_map:.4f})")

            if self.counter >= self.patience:
                self.stop_training = True
                print(f"Early stopping triggered: No improvement above {self.min_delta} threshold for {self.patience} epochs")
                if self.model:
                    self.model.request_early_stop()
                
    def set_model(self, model):
        """Set the model reference to call request_early_stop when needed"""
        self.model = model