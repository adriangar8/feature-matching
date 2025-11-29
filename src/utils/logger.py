"""
Logging Utilities

Provides unified logging interface supporting:
- Console output
- Weights & Biases (wandb)
- TensorBoard
- CSV files
"""

import os
import csv
import json
from datetime import datetime
from typing import Dict, Optional, Any
from pathlib import Path

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class Logger:
    """
    Unified logger supporting multiple backends.
    
    Args:
        project: Project name
        run_name: Run name (auto-generated if None)
        log_dir: Directory for local logs
        use_wandb: Enable Weights & Biases logging
        use_tensorboard: Enable TensorBoard logging
        config: Configuration dict to log
    """

    def __init__(
        self,
        project: str = "meta-feature-matching",
        run_name: Optional[str] = None,
        log_dir: str = "results/logs",
        use_wandb: bool = True,
        use_tensorboard: bool = False,
        config: Optional[Dict] = None,
    ):
        self.project = project
        self.run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(log_dir) / self.run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE

        # Initialize backends
        self.wandb_run = None
        self.tb_writer = None
        self.csv_file = None
        self.csv_writer = None
        self.csv_columns = None

        if self.use_wandb:
            self.wandb_run = wandb.init(
                project=project,
                name=self.run_name,
                config=config,
                reinit=True,
            )

        if self.use_tensorboard:
            self.tb_writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))

        # Save config
        if config:
            with open(self.log_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix: str = "",
    ) -> None:
        """
        Log metrics to all enabled backends.
        
        Args:
            metrics: Dict of metric names to values
            step: Current step/epoch
            prefix: Optional prefix for metric names
        """
        # Add prefix
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

        # Console
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print(f"[Step {step}] {metrics_str}")

        # Wandb
        if self.use_wandb and self.wandb_run:
            wandb.log(metrics, step=step)

        # TensorBoard
        if self.use_tensorboard and self.tb_writer:
            for k, v in metrics.items():
                self.tb_writer.add_scalar(k, v, step)

        # CSV
        self._log_to_csv(metrics, step)

    def _log_to_csv(self, metrics: Dict[str, float], step: int) -> None:
        """Log metrics to CSV file."""
        metrics_with_step = {"step": step, **metrics}

        # Initialize CSV on first call
        if self.csv_file is None:
            self.csv_file = open(self.log_dir / "metrics.csv", "w", newline="")
            self.csv_columns = list(metrics_with_step.keys())
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.csv_columns)
            self.csv_writer.writeheader()

        # Handle new columns
        new_cols = set(metrics_with_step.keys()) - set(self.csv_columns)
        if new_cols:
            # Reopen with new columns
            self.csv_file.close()
            self.csv_columns = list(set(self.csv_columns) | new_cols)

            # Read existing data
            existing_data = []
            csv_path = self.log_dir / "metrics.csv"
            if csv_path.exists():
                with open(csv_path, "r") as f:
                    reader = csv.DictReader(f)
                    existing_data = list(reader)

            # Rewrite with new columns
            self.csv_file = open(csv_path, "w", newline="")
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.csv_columns)
            self.csv_writer.writeheader()
            for row in existing_data:
                self.csv_writer.writerow(row)

        self.csv_writer.writerow(metrics_with_step)
        self.csv_file.flush()

    def log_image(
        self,
        name: str,
        image,
        step: int,
    ) -> None:
        """Log an image."""
        if self.use_wandb and self.wandb_run:
            wandb.log({name: wandb.Image(image)}, step=step)

        if self.use_tensorboard and self.tb_writer:
            # Assume image is a matplotlib figure or numpy array
            import matplotlib.pyplot as plt
            if isinstance(image, plt.Figure):
                self.tb_writer.add_figure(name, image, step)
            else:
                self.tb_writer.add_image(name, image, step, dataformats='HWC')

    def log_figure(
        self,
        name: str,
        figure,
        step: int,
        save_local: bool = True,
    ) -> None:
        """Log a matplotlib figure."""
        import matplotlib.pyplot as plt

        # Save locally
        if save_local:
            fig_dir = self.log_dir / "figures"
            fig_dir.mkdir(exist_ok=True)
            figure.savefig(fig_dir / f"{name}_{step}.png", dpi=150, bbox_inches='tight')

        # Log to backends
        if self.use_wandb and self.wandb_run:
            wandb.log({name: wandb.Image(figure)}, step=step)

        if self.use_tensorboard and self.tb_writer:
            self.tb_writer.add_figure(name, figure, step)

        plt.close(figure)

    def log_table(
        self,
        name: str,
        data: Dict[str, list],
        step: int,
    ) -> None:
        """Log a table of data."""
        if self.use_wandb and self.wandb_run:
            table = wandb.Table(
                columns=list(data.keys()),
                data=list(zip(*data.values())),
            )
            wandb.log({name: table}, step=step)

        # Save as CSV
        table_dir = self.log_dir / "tables"
        table_dir.mkdir(exist_ok=True)

        with open(table_dir / f"{name}_{step}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            writer.writeheader()
            for i in range(len(list(data.values())[0])):
                row = {k: v[i] for k, v in data.items()}
                writer.writerow(row)

    def save_model(
        self,
        model,
        name: str,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Save model checkpoint.
        
        Returns path to saved checkpoint.
        """
        import torch

        checkpoint_dir = self.log_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "metadata": metadata or {},
        }

        path = checkpoint_dir / f"{name}.pth"
        torch.save(checkpoint, path)

        return str(path)

    def finish(self) -> None:
        """Close all logging backends."""
        if self.csv_file:
            self.csv_file.close()

        if self.use_tensorboard and self.tb_writer:
            self.tb_writer.close()

        if self.use_wandb and self.wandb_run:
            wandb.finish()


# Convenience functions for backward compatibility
_global_logger: Optional[Logger] = None


def init_wandb(
    project: str = "meta-feature-matching",
    run_name: Optional[str] = None,
    config: Optional[Dict] = None,
) -> Any:
    """Initialize global logger with wandb."""
    global _global_logger
    _global_logger = Logger(
        project=project,
        run_name=run_name,
        config=config,
        use_wandb=True,
    )
    return wandb if WANDB_AVAILABLE else _global_logger


def log_metrics(split: str, metrics: Dict[str, float], step: int) -> None:
    """Log metrics using global logger."""
    if _global_logger:
        _global_logger.log_metrics(metrics, step, prefix=split)
    else:
        print(f"[{split}] Step {step}: {metrics}")


def get_logger() -> Optional[Logger]:
    """Get global logger instance."""
    return _global_logger
