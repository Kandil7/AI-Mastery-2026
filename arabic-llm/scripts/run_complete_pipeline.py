#!/usr/bin/env python3
"""
Balygh Complete Data Processing Pipeline

Runs the complete data processing pipeline for all 5 data sources:
1. Arabic Web Corpus
2. Extracted Books (8,424 Shamela books)
3. Metadata
4. Sanadset 368K Hadith Narrators
5. System Book Databases

Usage:
    python scripts/run_complete_pipeline.py [--audit] [--process] [--merge] [--train]
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

class PipelineConfig:
    """Pipeline configuration"""
    
    # Root paths
    ROOT = Path("K:/learning/technical/ai-ml/AI-Mastery-2026/arabic-llm")
    DATASETS_ROOT = Path("K:/learning/technical/ai-ml/AI-Mastery-2026/datasets")
    
    # Scripts
    SCRIPTS = ROOT / "scripts"
    
    # Output
    DATA_DIR = ROOT / "data"
    JSONL_DIR = DATA_DIR / "jsonl"
    EVAL_DIR = DATA_DIR / "evaluation"
    MODELS_DIR = ROOT / "models"
    
    # Target counts
    TARGET_EXAMPLES = 300000


# ============================================================================
# Pipeline Steps
# ============================================================================

def run_audit():
    """Step 1: Run complete data audit"""
    print("\n" + "=" * 70)
    print("STEP 1: Complete Data Audit")
    print("=" * 70)
    
    script = PipelineConfig.SCRIPTS / "complete_data_audit.py"
    
    if not script.exists():
        print(f"❌ Script not found: {script}")
        return False
    
    result = subprocess.run([sys.executable, str(script)], cwd=str(PipelineConfig.ROOT))
    
    if result.returncode == 0:
        print("\n✅ Data audit complete!")
        print(f"   Report: {PipelineConfig.DATA_DIR / 'complete_audit_report.json'}")
        return True
    else:
        print(f"\n❌ Data audit failed with code {result.returncode}")
        return False


def run_process_arabic_web():
    """Step 2a: Process Arabic Web Corpus"""
    print("\n" + "=" * 70)
    print("STEP 2a: Processing Arabic Web Corpus")
    print("=" * 70)
    
    script = PipelineConfig.SCRIPTS / "process_arabic_web.py"
    
    if not script.exists():
        print(f"❌ Script not found: {script}")
        return False
    
    result = subprocess.run([sys.executable, str(script)], cwd=str(PipelineConfig.ROOT))
    
    if result.returncode == 0:
        print("\n✅ Arabic web processing complete!")
        return True
    else:
        print(f"\n❌ Arabic web processing failed with code {result.returncode}")
        return False


def run_process_books():
    """Step 2b: Process Extracted Books"""
    print("\n" + "=" * 70)
    print("STEP 2b: Processing Extracted Books")
    print("=" * 70)
    
    script = PipelineConfig.SCRIPTS / "build_balygh_sft_dataset.py"
    
    if not script.exists():
        print(f"❌ Script not found: {script}")
        return False
    
    result = subprocess.run(
        [sys.executable, str(script), "--target-examples", "113000"],
        cwd=str(PipelineConfig.ROOT)
    )
    
    if result.returncode == 0:
        print("\n✅ Books processing complete!")
        return True
    else:
        print(f"\n❌ Books processing failed with code {result.returncode}")
        return False


def run_process_sanadset():
    """Step 2c: Process Sanadset"""
    print("\n" + "=" * 70)
    print("STEP 2c: Processing Sanadset Hadith Narrators")
    print("=" * 70)
    
    script = PipelineConfig.SCRIPTS / "process_sanadset.py"
    
    if not script.exists():
        print(f"❌ Script not found: {script}")
        return False
    
    result = subprocess.run([sys.executable, str(script)], cwd=str(PipelineConfig.ROOT))
    
    if result.returncode == 0:
        print("\n✅ Sanadset processing complete!")
        return True
    else:
        print(f"\n❌ Sanadset processing failed with code {result.returncode}")
        return False


def run_process_system_books():
    """Step 2d: Process System Books"""
    print("\n" + "=" * 70)
    print("STEP 2d: Processing System Book Databases")
    print("=" * 70)
    
    script = PipelineConfig.SCRIPTS / "integrate_datasets.py"
    
    if not script.exists():
        print(f"❌ Script not found: {script}")
        return False
    
    result = subprocess.run([sys.executable, str(script)], cwd=str(PipelineConfig.ROOT))
    
    if result.returncode == 0:
        print("\n✅ System books processing complete!")
        return True
    else:
        print(f"\n❌ System books processing failed with code {result.returncode}")
        return False


def run_refine():
    """Step 3: Refine with LLM"""
    print("\n" + "=" * 70)
    print("STEP 3: Refining with LLM")
    print("=" * 70)
    
    # Check for API key
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("⚠️  DEEPSEEK_API_KEY not set. Skipping LLM refinement.")
        print("   Set with: $env:DEEPSEEK_API_KEY=\"sk-...\"")
        return True  # Not critical, can skip
    
    script = PipelineConfig.SCRIPTS / "refine_balygh_sft_with_llm.py"
    
    if not script.exists():
        print(f"❌ Script not found: {script}")
        return False
    
    result = subprocess.run(
        [sys.executable, str(script), "--max-examples", "100000", "--provider", "deepseek"],
        cwd=str(PipelineConfig.ROOT),
        env={**os.environ, "DEEPSEEK_API_KEY": api_key}
    )
    
    if result.returncode == 0:
        print("\n✅ LLM refinement complete!")
        return True
    else:
        print(f"\n❌ LLM refinement failed with code {result.returncode}")
        return False


def run_merge():
    """Step 4: Merge all datasets"""
    print("\n" + "=" * 70)
    print("STEP 4: Merging All Datasets")
    print("=" * 70)
    
    # Create merge script inline
    merge_script = PipelineConfig.SCRIPTS / "merge_all_datasets.py"
    
    if not merge_script.exists():
        print("❌ Merge script not found")
        return False
    
    result = subprocess.run([sys.executable, str(merge_script)], cwd=str(PipelineConfig.ROOT))
    
    if result.returncode == 0:
        print("\n✅ Merge complete!")
        return True
    else:
        print(f"\n❌ Merge failed with code {result.returncode}")
        return False


def run_train():
    """Step 5: Train model"""
    print("\n" + "=" * 70)
    print("STEP 5: Training Model")
    print("=" * 70)
    
    script = PipelineConfig.SCRIPTS / "03_train_model.py"
    
    if not script.exists():
        print(f"❌ Script not found: {script}")
        return False
    
    config = PipelineConfig.ROOT / "configs" / "training_config.yaml"
    dataset = PipelineConfig.JSONL_DIR / "balygh_final_sft.jsonl"
    output = PipelineConfig.MODELS_DIR / "balygh-complete-v1"
    
    if not dataset.exists():
        print(f"❌ Dataset not found: {dataset}")
        print("   Run merge step first")
        return False
    
    result = subprocess.run(
        [
            sys.executable, str(script),
            "--config", str(config),
            "--dataset", str(dataset),
            "--output-dir", str(output)
        ],
        cwd=str(PipelineConfig.ROOT)
    )
    
    if result.returncode == 0:
        print("\n✅ Training complete!")
        print(f"   Model saved to: {output}")
        return True
    else:
        print(f"\n❌ Training failed with code {result.returncode}")
        return False


# ============================================================================
# Main Pipeline
# ============================================================================

def run_full_pipeline(args):
    """Run complete pipeline"""
    print("\n" + "=" * 70)
    print("Balygh Complete Data Processing Pipeline")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target: {PipelineConfig.TARGET_EXAMPLES:,} examples")
    print("=" * 70)
    
    steps = []
    
    # Step 1: Audit (always run)
    if args.audit or args.all:
        steps.append(("Data Audit", run_audit))
    
    # Step 2: Processing
    if args.process or args.all:
        steps.append(("Arabic Web", run_process_arabic_web))
        steps.append(("Extracted Books", run_process_books))
        steps.append(("Sanadset", run_process_sanadset))
        steps.append(("System Books", run_process_system_books))
    
    # Step 3: Refine (optional)
    if args.refine or args.all:
        steps.append(("LLM Refinement", run_refine))
    
    # Step 4: Merge (always if processing)
    if args.merge or args.all:
        steps.append(("Merge", run_merge))
    
    # Step 5: Train (optional)
    if args.train:
        steps.append(("Training", run_train))
    
    # Run steps
    successful = 0
    failed = 0
    
    for step_name, step_func in steps:
        try:
            if step_func():
                successful += 1
            else:
                failed += 1
                print(f"\n⚠️  Step '{step_name}' failed. Continue? (y/n)")
                if input().lower() != 'y':
                    break
        except KeyboardInterrupt:
            print(f"\n\n⚠️  Pipeline interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Step '{step_name}' error: {e}")
            failed += 1
            break
    
    # Summary
    print("\n" + "=" * 70)
    print("Pipeline Summary")
    print("=" * 70)
    print(f"Successful: {successful}/{len(steps)}")
    print(f"Failed: {failed}/{len(steps)}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    return failed == 0


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Balygh Complete Data Processing Pipeline"
    )
    
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Run data audit"
    )
    
    parser.add_argument(
        "--process",
        action="store_true",
        help="Process all data sources"
    )
    
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Refine with LLM"
    )
    
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge all datasets"
    )
    
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train model"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run complete pipeline (audit + process + refine + merge)"
    )
    
    args = parser.parse_args()
    
    # Default to all if no args
    if not any([args.audit, args.process, args.refine, args.merge, args.train, args.all]):
        args.all = True
    
    success = run_full_pipeline(args)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
