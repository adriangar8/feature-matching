#!/usr/bin/env python3
"""
Simplified benchmark runner script.
All logic has been refactored into modular components in src/.
"""

import argparse
import yaml
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.runner.benchmark_runner import run_benchmark


def main():
    parser = argparse.ArgumentParser(description="Feature Matching Benchmark (Final)")
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--quick", action="store_true", help="Quick test")
    parser.add_argument("--hpatches", type=str, help="HPatches root")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    args = parser.parse_args()
    
    # Default config (no MAML)
    config = {
        "hpatches_root": "dataset/hpatches",
        "seed": 42,
        "eval_traditional": False,
        "eval_deep": True,
        "eval_continual": True,
        "deep_epochs": 5,
        "epochs_source": 5,
        "epochs_target": 5,
        "lr": 5e-5,
        "batch_size": 32,
        "max_triplets": 50000,
        "max_pairs_per_seq": 1000,
        "max_distractors": 1000,
        "continual_methods": ["naive", "ewc", "lwf"],
        "ewc_lambda": 400,
        "lwf_lambda": 1.0,
        "output_dir": "results/paper_results",
    }
    
    if args.config:
        with open(args.config) as f:
            loaded = yaml.safe_load(f)
            if loaded:
                # Remove MAML if present
                loaded.pop("eval_maml", None)
                config.update(loaded)
    
    if args.hpatches:
        config["hpatches_root"] = args.hpatches
    
    if args.output_dir:
        config["output_dir"] = args.output_dir
    
    if args.quick:
        config.update({
            "deep_epochs": 5,
            "epochs_source": 5,
            "epochs_target": 5,
            "max_triplets": 5000,
            "max_pairs_per_seq": 20,
            "max_distractors": 50,
            "continual_methods": ["naive", "ewc"],
            "output_dir": "results/quick_test",
        })
    
    run_benchmark(config)


if __name__ == "__main__":
    main()
