import os
import argparse
from training.train_ddpm import train_ddpm
from training.train_ids import train_ids
from evaluation.benchmark import run_ablation_studies

def main():
    parser = argparse.ArgumentParser(description="Run full GAICS Dissertation Pipeline")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--skip-train-ddpm", action="store_true", help="Skip TabDDPM training")
    parser.add_argument("--skip-train-ids", action="store_true", help="Skip Attention IDS training")
    parser.add_argument("--skip-eval", action="store_true", help="Skip ablation studies")
    
    args = parser.parse_args()
    
    print("="*50)
    print("Starting 5-Layer Autonomous GenAI-Powered Cloud Defense Architecture")
    print("="*50)
    
    if not args.skip_train_ddpm:
        print("\n[Phase 1] Training Layer 1: Tabular DDPM")
        train_ddpm(args.config)
        
    if not args.skip_train_ids:
        print("\n[Phase 2] Training Layer 2: Attention IDS")
        train_ids(args.config)
        
    if not args.skip_eval:
        print("\n[Phase 3] Running Ablation Studies and Evaluation")
        run_ablation_studies(args.config)
        
    print("\nPipeline execution complete. You can now start the FastAPI server for the API deployment.")

if __name__ == "__main__":
    main()
