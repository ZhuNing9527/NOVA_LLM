import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend to non-interactive
import matplotlib
matplotlib.use('Agg')

# Set style for academic publications
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12

class DataValidator:
    """Validate and analyze the nutritional dataset."""

    def __init__(self, data_path: str):
        """Initialize validator with data path."""
        self.data_path = data_path
        self.df = None
        self.nutritional_features = [
            'Energy.(kcal)',
            'Protein.(g)',
            'Carbohydrate.(g)',
            'Sugars,.total.(g)',
            'Fiber,.total.dietary.(g)',
            'Total.Fat.(g)',
            'Fatty.acids,.total.saturated.(g)',
            'Sodium.(mg)',
            'Cholesterol.(mg)',
            'Calcium.(mg)',
            'Iron.(mg)',
            'Potassium.(mg)'
        ]

    def load_data(self):
        """Load the dataset."""
        print(f"Loading data from {self.data_path}...")
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {self.df.shape}")
        return self.df

    def basic_info(self):
        """Display basic dataset information."""
        print("\\n" + "="*60)
        print("BASIC DATASET INFORMATION")
        print("="*60)

        print(f"Total food items: {len(self.df)}")
        print(f"Total columns: {len(self.df.columns)}")
        print(f"Dataset size: {self.df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")

        # Check key columns
        key_columns = [
            'Food.code',
            'Main.food.description',
            'Combined.ingredients',
            'Processing.Method',
            'NOVA.Group'
        ]

        print(f"\\nKey columns check:")
        for col in key_columns:
            if col in self.df.columns:
                missing = self.df[col].isna().sum()
                print(f"  [OK] {col}: {len(self.df) - missing} values, {missing} missing")
            else:
                print(f"  [X] {col}: NOT FOUND")

    def nova_group_analysis(self, output_dir=None):
        """Analyze NOVA group distribution."""
        print("\\n" + "="*60)
        print("NOVA GROUP DISTRIBUTION")
        print("="*60)

        if 'NOVA.Group' not in self.df.columns:
            print("NOVA.Group column not found!")
            return

        nova_counts = self.df['NOVA.Group'].value_counts().sort_index()
        nova_percentages = (nova_counts / len(self.df) * 100).round(1)

        print("NOVA Group Distribution:")
        for group, count in nova_counts.items():
            percentage = nova_percentages[group]
            print(f"  {group}: {count:6d} items ({percentage:5.1f}%)")

        # Create plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(nova_counts)), nova_counts.values,
                     color=['#2E8B57', '#3CB371', '#FFD700', '#DC143C'])

        plt.xlabel('NOVA Group', fontsize=12)
        plt.ylabel('Number of Food Items', fontsize=12)
        plt.title('Distribution of Food Items Across NOVA Groups',
                 fontsize=14, fontweight='bold')
        plt.xticks(range(len(nova_counts)), nova_counts.index)
        plt.grid(True, alpha=0.3)

        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, nova_counts.values)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    f'{count}\\n({nova_percentages.iloc[i]}%)',
                    ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        if output_dir:
            plt.savefig(output_dir / 'nova_group_distribution.png', dpi=300, bbox_inches='tight')
        else:
            plt.savefig('nova_group_distribution.png', dpi=300, bbox_inches='tight')
        # plt.show()  # Commented out to avoid blocking in non-interactive mode

    def nutritional_features_analysis(self):
        """Analyze nutritional features."""
        print("\\n" + "="*60)
        print("NUTRITIONAL FEATURES ANALYSIS")
        print("="*60)

        # Check which nutritional features are available
        available_features = [f for f in self.nutritional_features if f in self.df.columns]
        missing_features = [f for f in self.nutritional_features if f not in self.df.columns]

        print(f"Available nutritional features: {len(available_features)}/{len(self.nutritional_features)}")
        if missing_features:
            print(f"Missing features: {missing_features}")

        if not available_features:
            print("No nutritional features found!")
            return

        # Basic statistics
        print(f"\\nNutritional Features Statistics:")
        stats_df = self.df[available_features].describe().round(2)

        # Display key statistics
        key_stats = ['count', 'mean', 'std', 'min', 'max']
        for feature in available_features:
            print(f"\\n  {feature}:")
            for stat in key_stats:
                if stat in stats_df.index:
                    value = stats_df.loc[stat, feature]
                    if pd.notna(value):
                        print(f"    {stat:8s}: {value:10.2f}")

        # Missing values analysis
        print(f"\\nMissing Values Analysis:")
        for feature in available_features:
            missing = self.df[feature].isna().sum()
            missing_pct = missing / len(self.df) * 100
            if missing > 0:
                print(f"  {feature}: {missing} ({missing_pct:.1f}%) missing")

    def data_quality_check(self):
        """Perform data quality checks."""
        print("\\n" + "="*60)
        print("DATA QUALITY CHECKS")
        print("="*60)

        # Check for duplicate food codes
        if 'Food.code' in self.df.columns:
            duplicates = self.df['Food.code'].duplicated().sum()
            print(f"Duplicate Food codes: {duplicates}")

        # Check for negative values in nutritional data
        print(f"\\nNegative Values Check:")
        for feature in self.nutritional_features:
            if feature in self.df.columns:
                negative_count = (self.df[feature] < 0).sum()
                if negative_count > 0:
                    print(f"  {feature}: {negative_count} negative values")

        # Check for extreme values (potential outliers)
        print(f"\\nExtreme Values Check (using IQR method):")
        for feature in self.nutritional_features:
            if feature in self.df.columns:
                Q1 = self.df[feature].quantile(0.25)
                Q3 = self.df[feature].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = ((self.df[feature] < lower_bound) |
                          (self.df[feature] > upper_bound)).sum()
                if outliers > 0:
                    print(f"  {feature}: {outliers} outliers detected")

        # Check ingredients and processing text fields
        text_fields = ['Combined.ingredients', 'Processing.Method', 'Main.food.description']
        print(f"\\nText Fields Quality:")
        for field in text_fields:
            if field in self.df.columns:
                empty_count = self.df[field].isna().sum() + (self.df[field] == '').sum()
                avg_length = self.df[field].astype(str).str.len().mean()
                print(f"  {field}:")
                print(f"    Empty/missing: {empty_count}")
                print(f"    Average length: {avg_length:.1f} characters")

    def create_correlation_matrix(self, output_dir=None):
        """Create correlation matrix for nutritional features."""
        print("\\n" + "="*60)
        print("NUTRITIONAL FEATURES CORRELATION ANALYSIS")
        print("="*60)

        available_features = [f for f in self.nutritional_features if f in self.df.columns]

        if len(available_features) < 2:
            print("Not enough nutritional features for correlation analysis!")
            return

        # Calculate correlation matrix
        corr_matrix = self.df[available_features].corr()

        # Create heatmap
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        sns.heatmap(corr_matrix,
                   mask=mask,
                   annot=True,
                   cmap='RdBu_r',
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar_kws={"shrink": .8},
                   annot_kws={"size": 8})

        plt.title('Correlation Matrix of Nutritional Features',
                 fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        if output_dir:
            plt.savefig(output_dir / 'nutritional_correlation_matrix.png', dpi=300, bbox_inches='tight')
        else:
            plt.savefig('nutritional_correlation_matrix.png', dpi=300, bbox_inches='tight')
        # plt.show()  # Commented out to avoid blocking in non-interactive mode

        # Print strongest correlations
        print(f"\\nStrongest Correlations (|r| > 0.7):")
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_val
                    ))

        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        for feat1, feat2, corr in corr_pairs:
            print(f"  {feat1} <-> {feat2}: {corr:.3f}")

    def generate_summary_report(self, save_path: str = None):
        """Generate a comprehensive summary report."""
        # Convert all numpy types to native Python types for JSON serialization
        report = {
            'dataset_info': {
                'total_items': int(len(self.df)),
                'total_columns': int(len(self.df.columns)),
                'file_size_mb': float(self.df.memory_usage(deep=True).sum() / 1024 / 1024)
            },
            'nova_distribution': {},
            'nutritional_features': {
                'available': int(len([f for f in self.nutritional_features if f in self.df.columns])),
                'total': int(len(self.nutritional_features))
            },
            'data_quality': {
                'duplicate_food_codes': int(self.df['Food.code'].duplicated().sum() if 'Food.code' in self.df.columns else 0),
                'missing_ingredients': int(self.df['Combined.ingredients'].isna().sum() if 'Combined.ingredients' in self.df.columns else 0),
                'missing_processing': int(self.df['Processing.Method'].isna().sum() if 'Processing.Method' in self.df.columns else 0)
            }
        }

        # Add NOVA distribution
        if 'NOVA.Group' in self.df.columns:
            nova_counts = self.df['NOVA.Group'].value_counts()
            # Convert numpy int64 to regular int for JSON serialization
            report['nova_distribution'] = {str(k): int(v) for k, v in nova_counts.to_dict().items()}

        # Save report
        if save_path:
            import json
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"\\nSummary report saved to: {save_path}")

        return report

def main():
    """Main execution function."""
    from pathlib import Path
    import sys

    # 构建数据文件路径
    base_path = Path(__file__).parent.parent
    data_path = base_path / 'data' / '副本FNDDS_2017_2018_NOVA_v3_nutrients.csv'
    output_dir = base_path / 'results'

    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if file exists
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        print("Please ensure the nutritional data file is in the data directory.")
        return

    # Initialize validator
    validator = DataValidator(str(data_path))

    # Load data
    validator.load_data()

    # Run all validations
    validator.basic_info()
    validator.nova_group_analysis(output_dir)
    validator.nutritional_features_analysis()
    validator.data_quality_check()
    validator.create_correlation_matrix(output_dir)

    # Generate summary report
    report_path = output_dir / 'data_validation_report.json'
    report = validator.generate_summary_report(str(report_path))

    print("\\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"[OK] Dataset loaded successfully: {report['dataset_info']['total_items']} items")
    print(f"[OK] NOVA groups analyzed: {len(report['nova_distribution'])} groups")
    print(f"[OK] Nutritional features: {report['nutritional_features']['available']}/{report['nutritional_features']['total']} available")
    print(f"[OK] Data quality checks completed")
    print(f"[OK] Visualization files generated:")
    print(f"  - nova_group_distribution.png")
    print(f"  - nutritional_correlation_matrix.png")
    print(f"[OK] Summary report: data_validation_report.json")

    print(f"\\nData is ready for nutritional assessment analysis!")

if __name__ == "__main__":
    main()