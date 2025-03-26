import numpy as np
import matplotlib.pyplot as plt

PLOT_FILE_NAME = "metrics_plot.png"

class MetricsPlotSink:
    """
    The MetricsPlotSink class records training metrics and saves them to a plot.

    Args:
        output_dir (str): Directory where the plot will be saved.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.history = []

    def update(self, values: dict):
        self.history.append(values)

    def save(self):
        if not self.history:
            print("No data to plot.")
            return

        def get_array(key):
            return np.array([h[key] for h in self.history if key in h])

        def safe_index(arr, idx):
            return arr[idx] if 0 <= idx < len(arr) else None

        epochs = get_array('epoch')
        train_loss = get_array('train_loss')
        test_loss = get_array('test_loss')
        test_coco_eval = [h['test_coco_eval_bbox'] for h in self.history if 'test_coco_eval_bbox' in h]
        ap = np.array([safe_index(x, 0) for x in test_coco_eval if x is not None], dtype=np.float32)
        ar = np.array([safe_index(x, 6) for x in test_coco_eval if x is not None], dtype=np.float32)
        ema_coco_eval = [h['ema_test_coco_eval_bbox'] for h in self.history if 'ema_test_coco_eval_bbox' in h]
        ema_ap = np.array([safe_index(x, 0) for x in ema_coco_eval if x is not None], dtype=np.float32)
        ema_ar = np.array([safe_index(x, 6) for x in ema_coco_eval if x is not None], dtype=np.float32)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        if len(epochs) > 0:
            if len(train_loss):
                axes[0].plot(epochs, train_loss, label='Training Loss', marker='o', linestyle='-')
            if len(test_loss):
                axes[0].plot(epochs, test_loss, label='Validation Loss', marker='o', linestyle='--')
            axes[0].set_title('Training and Validation Loss')
            axes[0].set_xlabel('Epoch Number')
            axes[0].set_ylabel('Loss Value')
            axes[0].legend()
            axes[0].grid(True)

        if ap.size > 0 or ema_ap.size > 0:
            if ap.size > 0:
                axes[1].plot(epochs[:len(ap)], ap, marker='o', linestyle='-', label='Base Model')
            if ema_ap.size > 0:
                axes[1].plot(epochs[:len(ema_ap)], ema_ap, marker='o', linestyle='--', label='EMA Model')
            axes[1].set_title('Average Precision @0.50:0.95')
            axes[1].set_xlabel('Epoch Number')
            axes[1].set_ylabel('AP')
            axes[1].legend()
            axes[1].grid(True)

        if ar.size > 0 or ema_ar.size > 0:
            if ar.size > 0:
                axes[2].plot(epochs[:len(ar)], ar, marker='o', linestyle='-', label='Base Model')
            if ema_ar.size > 0:
                axes[2].plot(epochs[:len(ema_ar)], ema_ar, marker='o', linestyle='--', label='EMA Model')
            axes[2].set_title('Average Recall @0.50:0.95')
            axes[2].set_xlabel('Epoch Number')
            axes[2].set_ylabel('AR')
            axes[2].legend()
            axes[2].grid(True)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{PLOT_FILE_NAME}")