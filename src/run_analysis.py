import argparse
import os
import sys
import json
import time
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed."""
    print("Checking dependencies...")

    required_packages = [
        'pandas', 'numpy', 'xgboost', 'sklearn',
        'matplotlib', 'seaborn', 'shap', 'requests'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"Missing packages: {missing_packages}")
        print("Please install with: pip install -r requirements.txt")
        return False

    print("All dependencies installed!")
    return True

def check_files():
    """Check if required files exist."""
    print("\nChecking required files...")

    from pathlib import Path
    base_path = Path(__file__).parent.parent

    required_files = [
        base_path / 'config.py',
        base_path / 'data' / '副本FNDDS_2017_2018_NOVA_v3_nutrients.csv',
        base_path / 'src' / 'generate_training_data.py',
        base_path / 'src' / 'train_proxy_model.py'
    ]

    missing_files = []

    for file in required_files:
        if not file.exists():
            missing_files.append(str(file))

    if missing_files:
        print(f"Missing files: {missing_files}")
        return False

    print("All required files found!")
    return True

def run_data_generation(sample_size=None):
    """Run the data generation step."""
    print("\n" + "="*60)
    print("STEP 1: GENERATING LLM TRAINING DATA")
    print("="*60)

    try:
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        from generate_training_data import NutritionalAssessmentGenerator
        from config import api_key
        import pandas as pd

        # 构建路径
        base_path = Path(__file__).parent.parent
        data_path = base_path / 'data' / '副本FNDDS_2017_2018_NOVA_v3_nutrients.csv'
        output_path = base_path / 'results' / 'nutritional_assessments.json'

        # Load data
        df = pd.read_csv(data_path)

        # Initialize generator
        generator = NutritionalAssessmentGenerator(api_key=api_key, delay=0.5)

        # Generate assessments

        if sample_size:
            print(f"Running with sample size: {sample_size}")
            generator.generate_batch_assessments(
                df=df,
                output_path=output_path,
                batch_size=20,
                sample_size=sample_size
            )
        else:
            print("Running with full dataset (this may take several hours)...")
            generator.generate_batch_assessments(
                df=df,
                output_path=output_path,
                batch_size=100
            )

        # Check if file was created
        if Path(output_path).exists():
            print("✓ Data generation completed successfully!")
            return True
        else:
            print("✗ Data generation failed!")
            return False

    except Exception as e:
        print(f"✗ Error in data generation: {e}")
        return False

def run_model_training():
    """Run the model training and validation step."""
    print("\n" + "="*60)
    print("STEP 2: TRAINING PROXY MODEL AND VALIDATION")
    print("="*60)

    try:
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        from train_proxy_model import ProxyModelTrainer

        # 构建路径
        base_path = Path(__file__).parent.parent
        assessment_data_path = base_path / 'results' / 'nutritional_assessments.json'
        nutritional_data_path = base_path / 'data' / '副本FNDDS_2017_2018_NOVA_v3_nutrients.csv'

        # Check if assessment data exists
        if not Path(assessment_data_path).exists():
            print(f"✗ Assessment data not found: {assessment_data_path}")
            print("Please run data generation first!")
            return False

        # Initialize trainer
        trainer = ProxyModelTrainer(assessment_data_path, nutritional_data_path)

        # Load and prepare data
        merged_df = trainer.load_and_merge_data()
        X, y = trainer.prepare_features(merged_df)

        # Train model
        results = trainer.train_xgboost_model(X, y)

        print(f"\nTraining Results:")
        print(f"  Test R² Score: {results['test_r2']:.4f}")
        print(f"  Test RMSE: {results['test_rmse']:.4f}")
        print(f"  Test MAE: {results['test_mae']:.4f}")

        # SHAP analysis
        trainer.calculate_shap_values()

        # Create output directory
        base_path = Path(__file__).parent.parent
        output_dir = base_path / 'results' / 'proxy_model_results'
        os.makedirs(output_dir, exist_ok=True)

        # Generate all plots
        trainer.create_shap_summary_plot(f"{output_dir}/shap_summary.png")
        trainer.create_shap_dependence_plots(output_dir)
        trainer.analyze_resolution_by_nova_group(merged_df, f"{output_dir}/nova_4_resolution.png")
        interaction_results = trainer.analyze_interaction_effects(merged_df, output_dir)
        trainer.create_model_performance_plots(results, output_dir)

        # Generate report
        report = trainer.generate_comprehensive_report(
            results, interaction_results, f"{output_dir}/validation_report.json"
        )

        print("✓ Model training and validation completed successfully!")
        print(f"✓ Results saved to: {output_dir}/")

        return True

    except Exception as e:
        print(f"✗ Error in model training: {e}")
        return False

def display_summary():
    """Display analysis summary and next steps."""
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)

    # Check for results
    from pathlib import Path
    base_path = Path(__file__).parent.parent
    assessment_file = base_path / 'results' / 'nutritional_assessments.json'
    results_dir = base_path / 'results' / 'proxy_model_results'

    if assessment_file.exists():
        try:
            with open(assessment_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            print(f"✓ Training Data Generated:")
            print(f"  - Total assessments: {data['metadata']['total_assessments']}")
            print(f"  - Success rate: {data['metadata']['success_rate']:.1%}")
            print(f"  - File size: {assessment_file.stat().st_size / 1024 / 1024:.1f} MB")
        except:
            print("✗ Could not read assessment data")

    if results_dir.exists():
        report_file = results_dir / "validation_report.json"
        if report_file.exists():
            try:
                with open(report_file, 'r', encoding='utf-8') as f:
                    report = json.load(f)

                validation = report['model_validation']
                print(f"\n✓ Model Validation Results:")
                print(f"  - R² Score: {validation['r2_score']:.4f}")
                print(f"  - RMSE: {validation['rmse']:.4f}")
                print(f"  - MAE: {validation['mae']:.4f}")
                print(f"  - Interpretation: {validation['interpretation']}")

                print(f"\n✓ Generated Files:")
                for file in results_dir.glob("*"):
                    print(f"  - {file.name}")

            except:
                print("✗ Could not read validation report")

    print(f"\nNext Steps:")
    print(f"1. Review the generated plots in 'proxy_model_results/'")
    print(f"2. Check the validation report for detailed analysis")
    print(f"3. Examine SHAP plots to understand feature importance")
    print(f"4. Review NOVA Group 4 resolution analysis")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run Complete Nutritional Assessment Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_analysis.py --sample-size 50    # Run with 50 samples (quick test)
  python run_analysis.py --full-dataset        # Run with full dataset (slow)
  python run_analysis.py                       # Interactive mode
        """
    )

    parser.add_argument(
        '--sample-size',
        type=int,
        help='Number of food samples to process (for testing)'
    )

    parser.add_argument(
        '--full-dataset',
        action='store_true',
        help='Process the full dataset (warning: very slow)'
    )

    args = parser.parse_args()

    print("Nutritional Assessment Framework - Complete Analysis Pipeline")
    print("=" * 60)

    # Check dependencies and files
    if not check_dependencies():
        sys.exit(1)

    if not check_files():
        sys.exit(1)

    # Determine analysis mode
    if args.full_dataset:
        sample_size = None
        print("Running in FULL DATASET mode")
    elif args.sample_size:
        sample_size = args.sample_size
        print(f"Running in SAMPLE mode with {sample_size} items")
    else:
        # Interactive mode
        print("\nChoose analysis mode:")
        print("1. Quick test (50 samples)")
        print("2. Medium test (500 samples)")
        print("3. Large test (2000 samples)")
        print("4. Full dataset (all samples)")

        choice = input("\nEnter choice (1-4): ").strip()

        if choice == "1":
            sample_size = 50
        elif choice == "2":
            sample_size = 500
        elif choice == "3":
            sample_size = 2000
        elif choice == "4":
            sample_size = None
            confirm = input("Continue? (y/N): ").strip().lower()
            if confirm != 'y':
                print("Cancelled.")
                return
        else:
            print("Invalid choice. Using quick test mode.")
            sample_size = 50

    # Record start time
    start_time = time.time()

    try:
        # Step 1: Generate training data
        if not run_data_generation(sample_size):
            print("Data generation failed. Exiting.")
            sys.exit(1)

        # Step 2: Train and validate model
        if not run_model_training():
            print("Model training failed. Exiting.")
            sys.exit(1)

        # Display summary
        display_summary()

        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)

        print(f"\n✓ Analysis completed in {hours:02d}:{minutes:02d}:{seconds:02d}")

    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()