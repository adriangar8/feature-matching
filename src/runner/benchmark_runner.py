"""
Main benchmark runner orchestrating all evaluations.
"""

from pathlib import Path
from typing import Dict
from dataclasses import asdict
from copy import deepcopy
import json
import torch
import wandb

from ..data.hpatches_manager import HPatchesManager
from ..data.structures import EvalPair
from ..descriptors.traditional import TraditionalExtractor
from ..evaluation.evaluator import evaluate_traditional, evaluate_deep
from ..training.trainer import train_model
from ..training.continual import train_continual
from ..visualization.tsne_viz import TSNEVisualizer
from ..visualization.matching_comparison import MatchingComparisonVisualizer
from ..visualization.methodology_figure import create_methodology_figure
from ..utils.config_utils import get_bool, get_int, get_float
from ..utils.seed_utils import set_seed
from ..utils.visualization import PatchVisualizer
from src.models.descriptor_wrapper import get_descriptor_model

def run_benchmark(config: Dict) -> Dict:
    """Run the complete benchmark."""
    
    # Initialize visualizer
    visualizer = PatchVisualizer(config.get("output_dir", "results/benchmark"))
    
    # Initialize wandb
    wandb.init(project="meta-feature-matching-final", config=config)
    
    seed = get_int(config, "seed", 42)
    set_seed(seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hpatches_root = config.get("hpatches_root", "/Data/adrian.garcia/hpatches/hpatches")
    
    data_mgr = HPatchesManager(hpatches_root, test_ratio=0.2, seed=seed)
    
    all_results = {
        "config": config,
        "seed": seed,
    }
    
    # Store models for T-SNE visualization
    tsne_models = {}
    
    # =========================================================================
    # Traditional Methods
    # =========================================================================
    if get_bool(config, "eval_traditional", True):
        print("\n" + "="*70)
        print("EVALUATING TRADITIONAL METHODS")
        print("="*70)
        
        all_results["traditional"] = {}
        
        for domain in ["illumination", "viewpoint"]:
            eval_pairs = data_mgr.create_eval_pairs(
                data_mgr.get_sequences(domain, "test"), 
                config.get("max_pairs_per_seq", 50)
            )
            
            all_results["traditional"][domain] = {}
            
            for method in ["sift", "orb", "brisk"]:
                print(f"\nEvaluating {method.upper()} on {domain}...")
                result = evaluate_traditional(
                    method, eval_pairs, 
                    max_distractors=config.get("max_distractors", 100), 
                    visualizer=visualizer, 
                    save_every=20,
                    hpatches_root=hpatches_root,
                )
                all_results["traditional"][domain][method] = asdict(result)
                wandb.log({f"traditional_{method}_{domain}_accuracy": result.accuracy})
                
                print(f"  Accuracy: {result.accuracy:.4f}")
                print(f"  Top-5 Accuracy: {result.accuracy_top5:.4f}")
                print(f"  Mean Rank: {result.mean_rank:.2f}")
        
        # Add SIFT to T-SNE models
        tsne_models["SIFT"] = (None, "sift")
    
    # =========================================================================
    # Deep Learning Methods
    # =========================================================================
    if get_bool(config, "eval_deep", True):
        print("\n" + "="*70)
        print("EVALUATING DEEP LEARNING METHODS")
        print("="*70)
        
        all_results["deep"] = {}
        
        for train_domain in ["illumination", "viewpoint"]:
            print(f"\nTraining on {train_domain}...")
            train_seqs = data_mgr.get_sequences(train_domain, "train")
            triplets = data_mgr.create_training_triplets(
                train_seqs, 
                config.get("max_triplets", 10000),
                min_negative_distance=50.0,
                use_hardest_negative=True,
            )
            
            val_seqs = data_mgr.get_sequences(train_domain, "test")
            val_pairs = data_mgr.create_eval_pairs(val_seqs, max_pairs_per_seq=50)
            
            model = get_descriptor_model("resnet50")
            model = train_model(
                model, triplets,
                epochs=get_int(config, "deep_epochs", 5),
                batch_size=get_int(config, "batch_size", 64),
                lr=get_float(config, "lr", 1e-4),
                device=device,
                log_prefix=f"deep_{train_domain}",
                val_pairs=val_pairs,
            )
            
            # Store for T-SNE
            tsne_models[f"ResNet50 ({train_domain[:5].capitalize()})"] = (deepcopy(model), "deep")
            
            all_results["deep"][f"resnet50_{train_domain}"] = {}
            
            for eval_domain in ["illumination", "viewpoint"]:
                print(f"\nEvaluating on {eval_domain}...")
                eval_pairs = data_mgr.create_eval_pairs(
                    data_mgr.get_sequences(eval_domain, "test"), 
                    config.get("max_pairs_per_seq", 50)
                )
                result = evaluate_deep(
                    model, eval_pairs, device, 
                    max_distractors=config.get("max_distractors", 100), 
                    visualizer=visualizer, 
                    method_name=f"resnet50_{train_domain}", 
                    save_every=20
                )
                all_results["deep"][f"resnet50_{train_domain}"][eval_domain] = asdict(result)
                wandb.log({f"deep_resnet50_{train_domain}_{eval_domain}_accuracy": result.accuracy})
                
                print(f"  Accuracy: {result.accuracy:.4f}")
                print(f"  Top-5 Accuracy: {result.accuracy_top5:.4f}")
                print(f"  Mean Rank: {result.mean_rank:.2f}")
    
    # =========================================================================
    # Continual Learning
    # =========================================================================
    if get_bool(config, "eval_continual", True):
        print("\n" + "="*70)
        print("EVALUATING CONTINUAL LEARNING METHODS")
        print("="*70)
        
        all_results["continual"] = {}
        
        for source_domain, target_domain in [("illumination", "viewpoint"), ("viewpoint", "illumination")]:
            transfer_key = f"{source_domain}_to_{target_domain}"
            print(f"\nContinual Learning: {transfer_key}")
            
            source_seqs = data_mgr.get_sequences(source_domain, "train")
            target_seqs = data_mgr.get_sequences(target_domain, "train")
            source_triplets = data_mgr.create_training_triplets(
                source_seqs, config.get("max_triplets", 10000),
                min_negative_distance=50.0, use_hardest_negative=True,
            )
            target_triplets = data_mgr.create_training_triplets(
                target_seqs, config.get("max_triplets", 10000),
                min_negative_distance=50.0, use_hardest_negative=True,
            )
            
            source_eval = data_mgr.create_eval_pairs(
                data_mgr.get_sequences(source_domain, "test"), 
                config.get("max_pairs_per_seq", 50)
            )
            target_eval = data_mgr.create_eval_pairs(
                data_mgr.get_sequences(target_domain, "test"), 
                config.get("max_pairs_per_seq", 50)
            )
            
            all_results["continual"][transfer_key] = {}
            
            for method in config.get("continual_methods", ["naive", "ewc", "lwf"]):
                print(f"\n  Method: {method}")
                
                print(f"  Training on source domain ({source_domain})...")
                model = get_descriptor_model("resnet50")
                model = train_model(
                    model, source_triplets,
                    epochs=get_int(config, "epochs_source", 5),
                    batch_size=get_int(config, "batch_size", 64),
                    lr=get_float(config, "lr", 1e-4),
                    device=device,
                    log_prefix=f"continual_{transfer_key}_{method}_source",
                )
                
                source_before = evaluate_deep(
                    model, source_eval, device, 
                    max_distractors=config.get("max_distractors", 100), 
                    visualizer=visualizer, 
                    method_name=f"continual_{method}_source_before", 
                    save_every=20
                )
                
                print(f"  Adapting to target domain ({target_domain})...")
                model = train_continual(
                    model, source_triplets, target_triplets,
                    method=method,
                    epochs_target=get_int(config, "epochs_target", 5),
                    batch_size=get_int(config, "batch_size", 64),
                    lr=get_float(config, "lr", 1e-4),
                    device=device,
                    log_prefix=f"continual_{transfer_key}",
                    ewc_lambda=get_float(config, "ewc_lambda", 400),
                    lwf_lambda=get_float(config, "lwf_lambda", 1.0),
                )
                
                source_after = evaluate_deep(
                    model, source_eval, device, 
                    max_distractors=config.get("max_distractors", 100), 
                    visualizer=visualizer, 
                    method_name=f"continual_{method}_source_after", 
                    save_every=20
                )
                target_after = evaluate_deep(
                    model, target_eval, device, 
                    max_distractors=config.get("max_distractors", 100), 
                    visualizer=visualizer, 
                    method_name=f"continual_{method}_target_after", 
                    save_every=20
                )
                
                forgetting = source_before.accuracy - source_after.accuracy
                forgetting_rate = forgetting / source_before.accuracy if source_before.accuracy > 0 else 0
                
                all_results["continual"][transfer_key][method] = {
                    "source_acc_before": source_before.accuracy,
                    "source_acc_after": source_after.accuracy,
                    "target_acc_after": target_after.accuracy,
                    "forgetting": forgetting,
                    "forgetting_rate": forgetting_rate,
                    "source_before_details": asdict(source_before),
                    "source_after_details": asdict(source_after),
                    "target_after_details": asdict(target_after),
                }
                
                print(f"    Source accuracy before: {source_before.accuracy:.4f}")
                print(f"    Source accuracy after:  {source_after.accuracy:.4f}")
                print(f"    Target accuracy after:  {target_after.accuracy:.4f}")
                print(f"    Forgetting rate:       {forgetting_rate*100:.1f}%")
                
                wandb.log({
                    f"continual_{transfer_key}_{method}_source_acc_before": source_before.accuracy,
                    f"continual_{transfer_key}_{method}_source_acc_after": source_after.accuracy,
                    f"continual_{transfer_key}_{method}_target_acc_after": target_after.accuracy,
                    f"continual_{transfer_key}_{method}_forgetting_rate": forgetting_rate * 100
                })
    
    # =========================================================================
    # Generate T-SNE Visualizations
    # =========================================================================
    print("\n" + "="*70)
    print("GENERATING T-SNE VISUALIZATIONS")
    print("="*70)
    
    output_dir = Path(config.get("output_dir", "results/benchmark"))
    tsne_dir = output_dir / "tsne_figures"
    tsne_dir.mkdir(parents=True, exist_ok=True)
    
    tsne_viz = TSNEVisualizer(tsne_dir)
    
    # Create samples for T-SNE
    tsne_samples = []
    illum_seqs = data_mgr.get_sequences("illumination", "test")[:2]
    view_seqs = data_mgr.get_sequences("viewpoint", "test")[:2]
    
    for seq in illum_seqs:
        sample = data_mgr.create_tsne_sample(seq, n_distractors=10)
        if sample:
            tsne_samples.append(sample)
    
    for seq in view_seqs:
        sample = data_mgr.create_tsne_sample(seq, n_distractors=10)
        if sample:
            tsne_samples.append(sample)
    
    if tsne_samples and tsne_models:
        # Create individual T-SNE figures
        for sample in tsne_samples:
            for model_name, (model, model_type) in tsne_models.items():
                if model_type == "sift":
                    embeddings = tsne_viz.extract_embeddings_sift(sample)
                else:
                    embeddings = tsne_viz.extract_embeddings_deep(model, sample, device)
                
                coords = tsne_viz.compute_tsne_2d(embeddings)
                
                safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
                tsne_viz.create_tsne_figure(
                    coords, sample, model_name,
                    tsne_dir / f"tsne_{safe_name}_{sample.domain}_{sample.seq_name}.png"
                )
                
                tsne_viz.create_global_context_figure(
                    sample, model_name,
                    tsne_dir / f"global_{safe_name}_{sample.domain}_{sample.seq_name}.png"
                )
        
        # Create combined comparison figure
        tsne_viz.create_combined_tsne_figure(
            tsne_samples, tsne_models, device,
            tsne_dir / "tsne_comparison_all_models.png"
        )
    
    # =========================================================================
    # Generate Qualitative Matching Comparisons
    # =========================================================================
    if get_bool(config, "generate_comparisons", True) and tsne_models:
        print("\\n" + "="*70)
        print("GENERATING MATCHING COMPARISONS (SIFT vs Learned)")
        print("="*70)
        
        comparison_dir = output_dir / "matching_comparisons"
        comparison_viz = MatchingComparisonVisualizer(comparison_dir)
        
        # Use the best learned model (viewpoint-trained for viewpoint, illumination-trained for illumination)
        for model_name, (model, model_type) in tsne_models.items():
            if model_type == "deep" and model is not None:
                comparison_viz.generate_comparisons(
                    data_mgr=data_mgr,
                    model=model,
                    device=device,
                    n_examples=3,
                    max_distractors=100,
                    model_name=model_name
                )
                break  # Just use one model for comparisons
    
    # =========================================================================
    # Generate Paper Figures
    # =========================================================================
    print("\n" + "="*70)
    print("GENERATING PAPER FIGURES")
    print("="*70)
    
    figures_dir = output_dir / "paper_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    create_methodology_figure(figures_dir)
    
    # =========================================================================
    # Generate Standard Visualizations
    # =========================================================================
    print("\n" + "="*70)
    print("GENERATING STANDARD VISUALIZATIONS")
    print("="*70)
    
    visualizer.create_all_figures(results=all_results)
    
    # Save results
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    summary = visualizer.get_summary()
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    print(f"Summary saved to: {summary_file}")
    
    # Print comprehensive summary
    print("\n" + "="*70)
    print("COMPREHENSIVE SUMMARY")
    print("="*70)
    
    if "traditional" in all_results:
        print("\nTraditional Methods:")
        print("  Method    | Illumination | Viewpoint")
        print("  ----------|--------------|----------")
        for method in ["sift", "orb", "brisk"]:
            illum = all_results["traditional"]["illumination"].get(method, {}).get("accuracy", 0)
            view = all_results["traditional"]["viewpoint"].get(method, {}).get("accuracy", 0)
            print(f"  {method.upper():<9} | {illum:.4f}       | {view:.4f}")
    
    if "deep" in all_results:
        print("\nDeep Learning Methods:")
        print("  Model           | Illumination | Viewpoint")
        print("  ----------------|--------------|-----------")
        for key, data in all_results["deep"].items():
            illum = data.get("illumination", {}).get("accuracy", 0)
            view = data.get("viewpoint", {}).get("accuracy", 0)
            print(f"  {key:<15} | {illum:.4f}       | {view:.4f}")
    
    if "continual" in all_results:
        print("\nContinual Learning:")
        print("  Transfer        | Method | Src Before | Src After | Target | Forgetting")
        print("  ----------------|--------|------------|-----------|--------|----------")
        for transfer, methods in all_results["continual"].items():
            for method, data in methods.items():
                src_b = data.get("source_acc_before", 0)
                src_a = data.get("source_acc_after", 0)
                tgt = data.get("target_acc_after", 0)
                forg = data.get("forgetting_rate", 0) * 100
                print(f"  {transfer[:15]:<15} | {method:<6} | {src_b:.4f}     | {src_a:.4f}    | {tgt:.4f} | {forg:5.1f}%")
    
    wandb.save(str(results_file))
    wandb.save(str(summary_file))
    wandb.finish()
    
    print("\n" + "="*70)
    print("BENCHMARK COMPLETED SUCCESSFULLY")
    print("="*70)
    
    return all_results

