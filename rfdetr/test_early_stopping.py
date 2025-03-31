import sys
import os
import time
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

# Add the project root to path so we can import the code
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rfdetr.main import Model, populate_args
from rfdetr.util.early_stopping import EarlyStoppingCallback

class MockModel:
    """Mock model that simulates the Model class but doesn't build a real model"""
    
    def __init__(self, map_values, **kwargs):
        """
        Args:
            map_values: List of mAP values to return for each epoch
            **kwargs: Arguments to pass to populate_args
        """
        self.map_values = map_values
        self.args = populate_args(**kwargs)
        self.stop_early = False
        self.current_epoch = 0
    
    def request_early_stop(self):
        """Same method as Model.request_early_stop"""
        self.stop_early = True
        print("Early stopping requested, will complete current epoch and stop")
    
    def train(self, callbacks=None, **kwargs):
        """Simulated train method that follows the same pattern as Model.train"""
        if callbacks is None:
            callbacks = defaultdict(list)
        
        # Set up the parameters
        args = populate_args(**kwargs)
        
        # We need a valid output directory for logs
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n===== Testing Early Stopping with Mock Model =====")
        print(f"Using map_values: {self.map_values}")
        if hasattr(args, 'early_stopping') and args.early_stopping:
            print(f"Early stopping params: patience={args.early_stopping_patience}, min_delta={args.early_stopping_min_delta}")
        
        print("\nStarting mock training...")
        start_time = time.time()
        
        for epoch in range(min(args.epochs, len(self.map_values))):
            self.current_epoch = epoch
            
            # Simulate one epoch of training
            epoch_start_time = time.time()
            time.sleep(0.2)  # To make output more readable
            
            # Generate mock train stats
            train_stats = {
                'loss': 1.0 / (epoch + 1),  # Decreasing loss
                'class_error': 0.5 / (epoch + 1)
            }
            
            # Generate mock evaluation stats with the pre-defined mAP
            map_value = self.map_values[epoch]
            test_stats = {
                'loss': 1.2 / (epoch + 1),
                'coco_eval_bbox': [map_value, map_value * 0.8, 0.0, 0.0, 0.0, 0.0, map_value * 0.9]
            }
            
            # Create log stats dictionary similar to the real train method
            log_stats = {
                'epoch': epoch,
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'n_parameters': 1000000  # Dummy value
            }
            
            if args.use_ema:
                # Add EMA metrics (slightly better than regular metrics)
                ema_map = map_value * 1.05
                log_stats['ema_test_coco_eval_bbox'] = [
                    ema_map, ema_map * 0.8, 0.0, 0.0, 0.0, 0.0, ema_map * 0.9
                ]
            
            print(f"Epoch {epoch}: mAP = {map_value:.4f}")
            
            # Write the log file similar to the real train method
            if args.output_dir:
                with (output_dir / "log.txt").open("a") as f:
                    f.write(f"{str(log_stats)}\n")
            
            # Call the on_fit_epoch_end callbacks
            for callback in callbacks["on_fit_epoch_end"]:
                callback(log_stats)
            
            # Check if early stopping was triggered
            if self.stop_early:
                print(f"\n✅ Early stopping triggered after epoch {epoch}")
                break
        else:
            print("\n❌ Early stopping was not triggered")
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds")

# Test scenarios with different mAP patterns

def test_scenario_1():
    """Steady improvement, no early stopping expected"""
    map_values = [0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48]
    model = MockModel(map_values=map_values, num_classes=2)
    
    # Initialize callbacks - this simulates what happens in detr.py
    callbacks = defaultdict(list)
    
    # Initialize early stopping callback - similar to how it would be done in detr.py
    early_stopping_callback = EarlyStoppingCallback(
        model=model,  # Pass model directly now
        patience=3,
        min_delta=0.005,
        use_ema=False
    )
    callbacks["on_fit_epoch_end"].append(early_stopping_callback.update)
    
    model.train(
        callbacks=callbacks, 
        epochs=10, 
        output_dir="test_output",
        early_stopping=True,
        early_stopping_patience=3,
        early_stopping_min_delta=0.005
    )

def test_scenario_2():
    """Early plateau, should trigger early stopping"""
    map_values = [0.30, 0.32, 0.34, 0.341, 0.342, 0.342, 0.343, 0.343, 0.344, 0.344]
    model = MockModel(map_values=map_values, num_classes=2)
    
    # Initialize callbacks
    callbacks = defaultdict(list)
    
    # Initialize early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        model=model,
        patience=3,
        min_delta=0.005,
        use_ema=False
    )
    callbacks["on_fit_epoch_end"].append(early_stopping_callback.update)
    
    model.train(
        callbacks=callbacks, 
        epochs=10, 
        output_dir="test_output",
        early_stopping=True,
        early_stopping_patience=3,
        early_stopping_min_delta=0.005
    )

def test_scenario_3():
    """Initial improvement then plateau"""
    map_values = [0.30, 0.35, 0.40, 0.45, 0.451, 0.452, 0.452, 0.453, 0.453, 0.454]
    model = MockModel(map_values=map_values, num_classes=2)
    
    # Initialize callbacks
    callbacks = defaultdict(list)
    
    # Initialize early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        model=model,
        patience=3,
        min_delta=0.005,
        use_ema=False
    )
    callbacks["on_fit_epoch_end"].append(early_stopping_callback.update)
    
    model.train(
        callbacks=callbacks, 
        epochs=10, 
        output_dir="test_output",
        early_stopping=True,
        early_stopping_patience=3,
        early_stopping_min_delta=0.005
    )

def test_scenario_4():
    """Decreasing performance"""
    map_values = [0.30, 0.32, 0.34, 0.33, 0.32, 0.31, 0.30, 0.29, 0.28, 0.27]
    model = MockModel(map_values=map_values, num_classes=2)
    
    # Initialize callbacks
    callbacks = defaultdict(list)
    
    # Initialize early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        model=model,
        patience=3,
        min_delta=0.005,
        use_ema=False
    )
    callbacks["on_fit_epoch_end"].append(early_stopping_callback.update)
    
    model.train(
        callbacks=callbacks, 
        epochs=10, 
        output_dir="test_output",
        early_stopping=True,
        early_stopping_patience=3,
        early_stopping_min_delta=0.005
    )

def test_scenario_5():
    """With EMA metrics"""
    map_values = [0.30, 0.32, 0.34, 0.341, 0.342, 0.342, 0.343, 0.343, 0.344, 0.344]
    model = MockModel(map_values=map_values, num_classes=2)
    
    # Initialize callbacks
    callbacks = defaultdict(list)
    
    # Initialize early stopping callback with EMA
    early_stopping_callback = EarlyStoppingCallback(
        model=model,
        patience=3,
        min_delta=0.005,
        use_ema=True
    )
    callbacks["on_fit_epoch_end"].append(early_stopping_callback.update)
    
    model.train(
        callbacks=callbacks, 
        epochs=10, 
        output_dir="test_output", 
        use_ema=True, 
        early_stopping=True,
        early_stopping_patience=3,
        early_stopping_min_delta=0.005,
        early_stopping_use_ema=True
    )

if __name__ == "__main__":
    # Make sure the output directory exists
    os.makedirs("test_output", exist_ok=True)
    
    print("\n\n🔍 SCENARIO 1: Steady improvement, no early stopping")
    test_scenario_1()
    
    print("\n\n🔍 SCENARIO 2: Early plateau, should trigger early stopping")
    test_scenario_2()
    
    print("\n\n🔍 SCENARIO 3: Initial improvement then plateau")
    test_scenario_3()
    
    print("\n\n🔍 SCENARIO 4: Decreasing performance")
    test_scenario_4()
    
    print("\n\n🔍 SCENARIO 5: Using EMA metrics")
    test_scenario_5()