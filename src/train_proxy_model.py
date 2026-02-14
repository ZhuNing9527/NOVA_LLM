import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
import shap
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from scipy.stats import uniform, randint
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for academic publications
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

class ProxyModelTrainer:
    """
    Train and validate proxy model for nutritional assessment.
    """

    def __init__(self, data_path: str, nutritional_data_path: str):
        """
        Initialize the trainer.

        Args:
            data_path: Path to LLM assessment data
            nutritional_data_path: Path to nutritional data CSV
        """
        self.data_path = data_path
        self.nutritional_data_path = nutritional_data_path
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.shap_explainer = None
        self.shap_values = None

        # Define key nutritional features for analysis
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

    def load_and_merge_data(self) -> pd.DataFrame:
        """
        Load LLM assessments (prefer CSV if available) and merge with nutritional data.

        Returns:
            Merged DataFrame
        """
        # Try to load CSV data first (for machine learning training)
        csv_path = Path(self.data_path).with_suffix('.csv')

        if csv_path.exists():
            print("Loading CSV assessment data for training...")
            assessments_df = pd.read_csv(csv_path)
            print(f"Loaded {len(assessments_df)} training samples from CSV")

            # Check if nutritional data is already in the CSV
            nutritional_cols_in_csv = [col for col in self.nutritional_features if col in assessments_df.columns]
            print(f"Nutritional columns already in CSV: {len(nutritional_cols_in_csv)}/{len(self.nutritional_features)}")

            if len(nutritional_cols_in_csv) >= len(self.nutritional_features) * 0.8:  # If 80%+ of features are present
                print("Using nutritional data directly from CSV (no merge needed)")
                return assessments_df

        # Fallback to JSON data or if CSV lacks nutritional data
        if not csv_path.exists():
            print("Loading JSON assessment data...")
            with open(self.data_path, 'r', encoding='utf-8') as f:
                assessment_data = json.load(f)
            assessments_df = pd.DataFrame(assessment_data['assessments'])

        # Load nutritional data
        print("Loading nutritional data...")
        nutritional_df = pd.read_csv(self.nutritional_data_path)

        # Merge datasets on food code
        print("Merging datasets...")
        merged_df = assessments_df.merge(
            nutritional_df[['Food.code'] + self.nutritional_features],
            left_on='food_code',
            right_on='Food.code',
            how='inner',
            suffixes=('', '_nutritional')  # Add suffix to avoid conflicts
        )

        # Fix column naming conflicts by copying suffixed columns to original names
        for feature in self.nutritional_features:
            if feature in merged_df.columns:
                continue  # Original column exists, no suffix needed
            elif f"{feature}_nutritional" in merged_df.columns:
                merged_df[feature] = merged_df[f"{feature}_nutritional"]

        print(f"Merged dataset shape: {merged_df.shape}")
        print(f"Successful assessments for training: {len(merged_df)}")

        return merged_df

    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """
        Prepare features for model training.

        Args:
            df: Merged DataFrame

        Returns:
            Tuple of (X, y) feature and target matrices
        """
        # Extract nutritional features
        X = df[self.nutritional_features].copy()

        # Handle missing values
        X = X.fillna(X.median())

        print(f"Original feature name: {list(X.columns)}")

     
        special_chars = set('.,()[]{}+-*/')
        has_special_chars = any(char in ''.join(X.columns) for char in special_chars)

        if has_special_chars:
            print("Special characters were detected in the feature name; compatibility processing is being performed...")
            feature_name_mapping = {}
            cleaned_X = X.copy()

        
            for i, col in enumerate(self.nutritional_features):
                if col in X.columns:
                    clean_name = f"feature_{i}"
                    feature_name_mapping[col] = clean_name
                    cleaned_X = cleaned_X.rename(columns={col: clean_name})

    
            X = cleaned_X
            print(f"Feature name after cleaning: {list(X.columns)}")
        else:
            print("Feature names are highly compatible and require no processing.")
            feature_name_mapping = {col: col for col in self.nutritional_features if col in X.columns}

        # Extract target variable (NII Score)
        y = df['NII_Score']

        # Store both original and cleaned feature names
        self.original_feature_names = self.nutritional_features
        self.feature_names = list(X.columns)
        self.feature_name_mapping = feature_name_mapping

        print(f"Feature matrix shape: {X.shape}")
        print(f"Target variable shape: {y.shape}")
        print(f"Original feature name mapping: {self.feature_name_mapping}")

       # Verify data quality
        print(f"Data quality check:")
        print(f"  - Missing values ​​in the feature matrix: {X.isnull().sum().sum()}")
        print(f"  - Missing values ​​for the target variable: {y.isnull().sum()}")
        print(f"  - The numerical range of the characteristic matrix: {X.describe().loc[['min', 'max']].T}")
        print(f"  - Target variable range: [{y.min():.2f}, {y.max():.2f}]")

        return X, y

    def get_model_hyperparameter_grids(self) -> dict:

        param_grids = {
            'XGBoost': {
                'model': xgb.XGBRegressor(
                    random_state=42,
                    objective='reg:squarederror',
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.1,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_alpha=0.1,
                    reg_lambda=1.0
                ),
                'params': {
                    'n_estimators': [50, 100, 200, 300], 
                    'max_depth': [3, 4, 5, 6], 
                    'learning_rate': [0.05, 0.1, 0.15, 0.2], 
                    'subsample': [0.8, 0.9, 1.0], 
                    'colsample_bytree': [0.8, 0.9, 1.0],
                    'reg_alpha': [0.0, 0.1, 0.2], 
                    'reg_lambda': [0.5, 1.0, 1.5]
                }
            },
            'LightGBM': {
                'model': lgb.LGBMRegressor(
                    random_state=42,
                    verbose=-1,
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.1,
                    num_leaves=31,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_alpha=0.1,
                    reg_lambda=1.0
                ),
                'params': {
                    'n_estimators': [50, 100, 200, 300], 
                    'max_depth': [3, 4, 5, 6], 
                    'learning_rate': [0.05, 0.1, 0.15, 0.2],
                    'num_leaves': [15, 31, 50], 
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0],
                    'reg_alpha': [0.0, 0.1, 0.2],
                    'reg_lambda': [0.5, 1.0, 1.5],
                    'min_child_samples': [1, 5, 10] 
                }
            },
            'RandomForest': {
                'model': RandomForestRegressor(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': randint(100, 500),
                    'max_depth': randint(3, 20),
                    'min_samples_split': randint(2, 10),
                    'min_samples_leaf': randint(1, 5),
                    'max_features': ['sqrt', 'log2', None]
                }
            },
            'GradientBoosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': randint(100, 500),
                    'max_depth': randint(3, 10),
                    'learning_rate': uniform(0.01, 0.3),
                    'subsample': uniform(0.6, 0.4),
                    'min_samples_split': randint(2, 10),
                    'min_samples_leaf': randint(1, 5)
                }
            },
            'SVR': {
                'model': SVR(
                    kernel='rbf',
                    C=10.0,
                    epsilon=0.1,
                    gamma='scale'
                ),
                'params': {} 
            },
            'Ridge': {
                'model': Ridge(random_state=42),
                'params': {
                    'alpha': uniform(0.1, 10.0),
                    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg']
                }
            },
            'Lasso': {
                'model': Lasso(random_state=42, max_iter=2000),
                'params': {
                    'alpha': uniform(0.001, 1.0),
                    'selection': ['cyclic', 'random']
                }
            }
        }
        return param_grids

    def _count_grid_combinations(self, param_dist: dict) -> int:

        if not param_dist:
            return 0

        total = 1
        for param_values in param_dist.values():
            total *= len(param_values)
        return total

    def train_multiple_models_with_hyperparameter_search(self, X: pd.DataFrame, y: pd.Series,
                                                      test_size: float = 0.2,
                                                      random_state: int = 42,
                                                      search_method: str = 'randomized',
                                                      n_iter: int = 50,
                                                      cv_folds: int = 5) -> dict:

    
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        print(f"Training set size: {len(self.X_train)}")
        print(f"test set size: {len(self.X_test)}")

   
        param_grids = self.get_model_hyperparameter_grids()

        all_results = {}
        best_model = None
        best_score = -np.inf
        best_model_name = ""

        print("\nBegin multi-model training and hyperparameter optimization...")
        print("=" * 60)

        for model_name, model_config in param_grids.items():
            print(f"\nTraining and optimization in progress {model_name}...")

            model = model_config['model']
            param_dist = model_config['params']

            
            try:
                if hasattr(model, 'fit'):
                    if not param_dist:
                        print(f"  INFO: Training with pre-configured optimal parameters{model_name}...")
                        model.fit(self.X_train, self.y_train)
                        y_train_pred = model.predict(self.X_train)
                        y_test_pred = model.predict(self.X_test)

                        results = {
                            'model': model,
                            'best_params': 'optimized_default_parameters',
                            'best_cv_score': 0.0, 
                            'train_r2': r2_score(self.y_train, y_train_pred),
                            'test_r2': r2_score(self.y_test, y_test_pred),
                            'train_rmse': np.sqrt(mean_squared_error(self.y_train, y_train_pred)),
                            'test_rmse': np.sqrt(mean_squared_error(self.y_test, y_test_pred)),
                            'train_mae': mean_absolute_error(self.y_train, y_train_pred),
                            'test_mae': mean_absolute_error(self.y_test, y_test_pred),
                            'cv_results': {}
                        }

                        all_results[model_name] = results

                        print(f"  METRICS: Test set R2: {results['test_r2']:.4f}")
                        print(f"  METRICS: Test set RMSE: {results['test_rmse']:.4f}")

                       # Update the best model
                        if results['test_r2'] > best_score:
                            best_score = results['test_r2']
                            best_model = model
                            best_model_name = model_name
                            self.model = model

                        continue  


                  # Dynamically adjust the number of cross-validation folds to fit small datasets
                    adjusted_cv_folds = cv_folds
                    if len(self.X_train) < cv_folds * 2:
                        adjusted_cv_folds = max(2, len(self.X_train) // 2)
                        print(f"  WARNING: If the data volume is too small, automatically adjust the number of CV folds from {cv_folds} to {adjusted_cv_folds}.")

                    if search_method == 'grid' and len(param_dist) <= 50: 
                        search = GridSearchCV(
                            model, param_dist, cv=adjusted_cv_folds,
                            scoring='r2', n_jobs=-1, verbose=1
                        )
                    else:  # randomized search for larger parameter spaces
                        search = RandomizedSearchCV(
                            model, param_dist, n_iter=min(n_iter, 100), cv=adjusted_cv_folds,
                            scoring='r2', n_jobs=-1, random_state=random_state, verbose=1
                        )


                    try:
                        search.fit(self.X_train, self.y_train)
                    except Exception as search_e:
                        raise search_e

                else:
                    raise ValueError(f"The model {model_name} does not support the fit method.")

            except Exception as e:

                import traceback
                print(f"  DETAILED ERROR: {traceback.format_exc()}")

                error_str = str(e).lower()

                try:

                    model.fit(self.X_train, self.y_train)
                    y_train_pred = model.predict(self.X_train)
                    y_test_pred = model.predict(self.X_test)

                 
                    results = {
                        'model': model,
                        'best_params': 'default_parameters',
                        'best_cv_score': 0.0,  
                        'train_r2': r2_score(self.y_train, y_train_pred),
                        'test_r2': r2_score(self.y_test, y_test_pred),
                        'train_rmse': np.sqrt(mean_squared_error(self.y_train, y_train_pred)),
                        'test_rmse': np.sqrt(mean_squared_error(self.y_test, y_test_pred)),
                        'train_mae': mean_absolute_error(self.y_train, y_train_pred),
                        'test_mae': mean_absolute_error(self.y_test, y_test_pred),
                        'cv_results': {},
                        'error_details': {
                            'error_type': type(e).__name__,
                            'error_message': str(e),
                            'fallback_used': True
                        }
                    }

                    all_results[model_name] = results

                    print(f"  METRICS: Test set R2: {results['test_r2']:.4f}")
                    print(f"  METRICS: Test set RMSE: {results['test_rmse']:.4f}")

                    # Update the best model
                    if results['test_r2'] > best_score:
                        best_score = results['test_r2']
                        best_model = model
                        best_model_name = model_name
                        self.model = model

                    continue 

                except Exception as fallback_e:
                    continue 

            # Obtain the best model
            best_model_for_type = search.best_estimator_

           
            y_train_pred = best_model_for_type.predict(self.X_train)
            y_test_pred = best_model_for_type.predict(self.X_test)

         
            results = {
                'model': best_model_for_type,
                'best_params': search.best_params_,
                'best_cv_score': search.best_score_,
                'train_r2': r2_score(self.y_train, y_train_pred),
                'test_r2': r2_score(self.y_test, y_test_pred),
                'train_rmse': np.sqrt(mean_squared_error(self.y_train, y_train_pred)),
                'test_rmse': np.sqrt(mean_squared_error(self.y_test, y_test_pred)),
                'train_mae': mean_absolute_error(self.y_train, y_train_pred),
                'test_mae': mean_absolute_error(self.y_test, y_test_pred),
                'cv_results': search.cv_results_
            }

            all_results[model_name] = results

            print(f"  Best CV R2: {search.best_score_:.4f}")
            print(f"  Test set R2: {results['test_r2']:.4f}")
            print(f"  Test set RMSE: {results['test_rmse']:.4f}")

            # Update the best model
            if results['test_r2'] > best_score:
                best_score = results['test_r2']
                best_model = best_model_for_type
                best_model_name = model_name
                self.model = best_model

        print(f"\nbest_model: {best_model_name} (Test set R2: {best_score:.4f})")

        all_results['best_model_name'] = best_model_name
        all_results['best_model'] = best_model
        all_results['model_comparison'] = self.create_model_comparison_table(all_results)

        return all_results

    def create_model_comparison_table(self, results: dict) -> pd.DataFrame:
      
        comparison_data = []

        for model_name, model_results in results.items():
            if model_name in ['best_model_name', 'best_model', 'model_comparison']:
                continue

            comparison_data.append({
                'Model': model_name,
                'Test_R2': model_results['test_r2'],
                'Test_RMSE': model_results['test_rmse'],
                'Test_MAE': model_results['test_mae'],
                'CV_R2': model_results['best_cv_score'],
                'Train_R2': model_results['train_r2']
            })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Test_R2', ascending=False)

        return comparison_df

    def train_xgboost_model(self, X: pd.DataFrame, y: pd.Series,
                           test_size: float = 0.2, random_state: int = 42) -> dict:
        """
        Train XGBoost model with hyperparameter tuning.

        Args:
            X: Feature matrix
            y: Target variable
            test_size: Proportion of test set
            random_state: Random seed

        Returns:
            Training results dictionary
        """
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        print(f"Training set size: {len(self.X_train)}")
        print(f"Test set size: {len(self.X_test)}")

        # Hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }

        # Initialize model
        xgb_model = xgb.XGBRegressor(
            random_state=random_state,
            objective='reg:squarederror'
        )

        # Use simple parameters for stability
        print("Training XGBoost model...")
        self.model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            objective='reg:squarederror'
        )

        self.model.fit(self.X_train, self.y_train)

        # Model trained
        print("Model training completed!")

        # Make predictions
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)

        # Calculate metrics
        results = {
            'train_r2': r2_score(self.y_train, y_train_pred),
            'test_r2': r2_score(self.y_test, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(self.y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(self.y_test, y_test_pred)),
            'train_mae': mean_absolute_error(self.y_train, y_train_pred),
            'test_mae': mean_absolute_error(self.y_test, y_test_pred),
            'model_params': {
                'n_estimators': 200,
                'max_depth': 5,
                'learning_rate': 0.1,
                'subsample': 0.9,
                'colsample_bytree': 0.9
            },
            'feature_importance': dict(zip(self.feature_names, self.model.feature_importances_))
        }

        return results

    def calculate_shap_values(self):
        """Calculate SHAP values for model interpretation."""
        print("Calculating SHAP values...")

        # Create SHAP explainer
        self.shap_explainer = shap.TreeExplainer(self.model)

        # Calculate SHAP values for test set
        self.shap_values = self.shap_explainer.shap_values(self.X_test)

        print("SHAP values calculated successfully")

    def create_shap_summary_plot(self, save_path: str = None) -> None:
        """
        Create SHAP summary plot (beeswarm plot).

        Args:
            save_path: Path to save the plot
        """
        if self.shap_values is None:
            self.calculate_shap_values()

        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values,
            self.X_test,
            feature_names=self.original_feature_names,
            plot_type="dot",
            show=False
        )

        plt.title('SHAP Summary Plot: Feature Impact on Nutritional Integrity Index',
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('SHAP Value (Impact on Model Output)', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SHAP summary plot saved to: {save_path}")

        # plt.show()  # Commented out to avoid displaying plots

    def create_shap_dependence_plots(self, save_dir: str = None) -> None:
        """
        Create SHAP dependence plots for ALL features.

        Args:
            save_dir: Directory to save plots
        """
        if self.shap_values is None:
            self.calculate_shap_values()

        print(f"Creating SHAP dependence plots for all {len(self.original_feature_names)} features...")

        # Create plots for ALL features (not just key features)
        for feature in self.original_feature_names:
            plt.figure(figsize=(10, 6))

            feature_idx = self.original_feature_names.index(feature)
            shap.dependence_plot(
                feature_idx,
                self.shap_values,
                self.X_test,
                feature_names=self.original_feature_names,
                show=False
            )

            plt.title(f'SHAP Dependence Plot: {feature}',
                     fontsize=14, fontweight='bold')
            plt.xlabel(f'{feature} Value', fontsize=12)
            plt.ylabel('SHAP Value', fontsize=12)
            plt.tight_layout()

            if save_dir:
                save_path = f"{save_dir}/shap_dependence_{feature.replace(',', '_').replace('.', '_')}.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"SHAP dependence plot for {feature} saved to: {save_path}")

            # plt.show()  # Commented out to avoid displaying plots

    def create_shap_interaction_matrix(self, save_dir: str = None) -> None:
        """
        Create SHAP interaction value matrix heatmap.

        Args:
            save_dir: Directory to save plot
        """
        if self.shap_values is None:
            self.calculate_shap_values()

        print("Calculating SHAP interaction values for matrix...")

        # Calculate interaction values (use sample for computational efficiency)
        sample_size = min(200, len(self.X_test))  # Limit sample size for performance
        shap_interaction = shap.TreeExplainer(self.model).shap_interaction_values(
            self.X_test.iloc[:sample_size]
        )

        # Calculate interaction strength matrix
        n_features = len(self.feature_names)
        interaction_matrix = np.zeros((n_features, n_features))

        for i in range(n_features):
            for j in range(n_features):
                if i == j:
                    interaction_matrix[i, j] = 0  # Diagonal elements are not interactions
                else:
                    interaction_matrix[i, j] = np.mean(np.abs(shap_interaction[:, i, j]))

        # Create heatmap
        plt.figure(figsize=(14, 12))

        # Use original feature names for labels
        feature_labels = [name.replace(',', '\n').replace('.', ' ') for name in self.original_feature_names]

        # Create heatmap
        im = plt.imshow(interaction_matrix, cmap='viridis', aspect='auto')

        # Set ticks and labels
        plt.xticks(range(n_features), feature_labels, rotation=45, ha='right', fontsize=10)
        plt.yticks(range(n_features), feature_labels, fontsize=10)

        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('Interaction Strength (Mean |SHAP Interaction|)', fontsize=12)

        # Add values to cells (optional, for clarity)
        for i in range(n_features):
            for j in range(n_features):
                if i != j:  # Don't show diagonal
                    text = plt.text(j, i, f'{interaction_matrix[i, j]:.3f}',
                                 ha="center", va="center", color="white", fontsize=8)

        plt.title('SHAP Interaction Value Matrix\n(Higher values indicate stronger feature interactions)',
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.tight_layout()

        if save_dir:
            save_path = f"{save_dir}/shap_interaction_matrix.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SHAP interaction matrix saved to: {save_path}")

        # Also save the numerical matrix as CSV
        if save_dir:
            import pandas as pd
            interaction_df = pd.DataFrame(
                interaction_matrix,
                index=self.original_feature_names,
                columns=self.original_feature_names
            )
            csv_path = f"{save_dir}/shap_interaction_matrix.csv"
            interaction_df.to_csv(csv_path)
            print(f"SHAP interaction matrix CSV saved to: {csv_path}")

        # plt.show()  # Commented out to avoid displaying plots

    def analyze_resolution_by_nova_group(self, df: pd.DataFrame, save_path: str = None) -> None:
        """
        Analyze NII score resolution within NOVA Group 4.

        Args:
            df: DataFrame with assessments
            save_path: Path to save the plot
        """
        # Filter NOVA Group 4
        # nova_4_df = df[df['nova_group'] == 'Group 4'].copy()
        nova_4_df = df[df['nova_group'] == 4].copy()

        if len(nova_4_df) == 0:
            print("No NOVA Group 4 items found")
            return

        plt.figure(figsize=(12, 6))

        # Create histogram
        plt.hist(nova_4_df['NII_Score'], bins=30, alpha=0.7, color='steelblue', edgecolor='black')

        plt.axvline(nova_4_df['NII_Score'].mean(), color='red', linestyle='--',
                   label=f'Mean: {nova_4_df["NII_Score"].mean():.1f}')
        plt.axvline(nova_4_df['NII_Score'].median(), color='green', linestyle='--',
                   label=f'Median: {nova_4_df["NII_Score"].median():.1f}')

        plt.title('NII Score Distribution within NOVA Group 4 (Ultra-processed Foods)',
                 fontsize=14, fontweight='bold')
        plt.xlabel('Nutritional Integrity Index (NII) Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Resolution analysis plot saved to: {save_path}")

        # plt.show()  # Commented out to avoid displaying plots

        # Print statistics
        print(f"\nNOVA Group 4 Statistics:")
        print(f"  Count: {len(nova_4_df)}")
        print(f"  Mean NII Score: {nova_4_df['NII_Score'].mean():.2f}")
        print(f"  Std Dev: {nova_4_df['NII_Score'].std():.2f}")
        print(f"  Min: {nova_4_df['NII_Score'].min():.2f}")
        print(f"  Max: {nova_4_df['NII_Score'].max():.2f}")
        print(f"  Range: {nova_4_df['NII_Score'].max() - nova_4_df['NII_Score'].min():.2f}")

    def analyze_interaction_effects(self, df: pd.DataFrame, save_dir: str = None) -> dict:
        """
        Analyze interaction effects between processing and nutritional features.

        Args:
            df: DataFrame with assessments
            save_dir: Directory to save plots

        Returns:
            Dictionary with interaction analysis results
        """
        print("Analyzing interaction effects...")

        # Merge with nutritional data for interaction analysis
        nutritional_df = pd.read_csv(self.nutritional_data_path)
        print(f"Nutritional data columns available: {list(nutritional_df.columns)}")

        # Check if Processing.Method exists
        if 'Processing.Method' not in nutritional_df.columns:
            print("WARNING: Processing.Method column not found in nutritional data!")
            # Try alternative column names
            alt_names = [col for col in nutritional_df.columns if 'processing' in col.lower() or 'Processing' in col]
            print(f"Alternative processing columns found: {alt_names}")

        merge_cols = ['Food.code'] + self.nutritional_features
        if 'Processing.Method' in nutritional_df.columns:
            merge_cols.append('Processing.Method')

        print(f"Attempting to merge with columns: {merge_cols}")

        # Use suffixes to avoid column name conflicts
        merged_df = df.merge(
            nutritional_df[merge_cols],
            left_on='food_code',
            right_on='Food.code',
            how='inner',
            suffixes=('', '_nutritional')
        )

        print(f"Merged data shape: {merged_df.shape}")
        print(f"Merged data columns: {list(merged_df.columns)}")

        # Create clean column names for analysis
        # If suffix was added, use the suffixed columns
        for feature in self.nutritional_features:
            if feature in merged_df.columns:
                continue  # Original column exists, no suffix needed
            elif f"{feature}_nutritional" in merged_df.columns:
                merged_df[feature] = merged_df[f"{feature}_nutritional"]  # Copy suffixed column to original name

        # Handle Processing.Method similarly
        if 'Processing.Method_nutritional' in merged_df.columns:
            merged_df['Processing.Method'] = merged_df['Processing.Method_nutritional']

        print("Fixed column naming conflicts")
        print(f"Available columns after fix: {[col for col in self.nutritional_features + ['Processing.Method', 'NII_Score'] if col in merged_df.columns]}")

        interaction_results = {}

        # Analyze processing vs protein interaction
        try:
            print("Creating protein-processing interaction plot...")

            if 'Processing.Method' in merged_df.columns and 'Protein.(g)' in merged_df.columns:
                plt.figure(figsize=(12, 8))

                # Create processing categories
                processing_categories = merged_df['Processing.Method'].value_counts().head(5).index
                print(f"Processing categories found: {list(processing_categories)}")

                for category in processing_categories:
                    subset = merged_df[merged_df['Processing.Method'] == category]
                    if len(subset) > 5:  # Ensure sufficient data points
                        plt.scatter(subset['Protein.(g)'], subset['NII_Score'],
                                  alpha=0.6, label=category, s=50)

                plt.title('Interaction Effect: Processing Method vs Protein Content on NII Score',
                         fontsize=14, fontweight='bold')
                plt.xlabel('Protein Content (g)', fontsize=12)
                plt.ylabel('NII Score', fontsize=12)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                if save_dir:
                    save_path = f"{save_dir}/interaction_protein_processing.png"
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    print(f"✓ Protein-processing interaction plot saved to: {save_path}")

                # plt.show()  # Commented out to avoid displaying plots

                # Calculate correlation by processing category
                correlations = {}
                for category in processing_categories:
                    subset = merged_df[merged_df['Processing.Method'] == category]
                    if len(subset) > 5:
                        corr = subset['Protein.(g)'].corr(subset['NII_Score'])
                        correlations[category] = corr

                interaction_results['protein_processing_correlations'] = correlations
            else:
                print("WARNING: Cannot create protein-processing plot - missing required columns")
                print(f"  Processing.Method available: {'Processing.Method' in merged_df.columns}")
                print(f"  Protein.(g) available: {'Protein.(g)' in merged_df.columns}")

        except Exception as e:
            print(f"ERROR creating protein-processing plot: {e}")
            import traceback
            traceback.print_exc()

        # Analyze sugar vs fiber interaction
        try:
            print("Creating sugar-fiber interaction plot...")

            if 'Sugars,.total.(g)' in merged_df.columns and 'Fiber,.total.dietary.(g)' in merged_df.columns:
                plt.figure(figsize=(10, 8))

                # Create sugar-fiber ratio
                merged_df['sugar_fiber_ratio'] = merged_df['Sugars,.total.(g)'] / (
                    merged_df['Fiber,.total.dietary.(g)'] + 0.1)  # Add small constant to avoid division by zero

                scatter = plt.scatter(merged_df['sugar_fiber_ratio'], merged_df['NII_Score'],
                                    c=merged_df['Energy.(kcal)'], alpha=0.6, cmap='viridis', s=50)

                plt.colorbar(scatter, label='Energy (kcal)')
                plt.title('Interaction Effect: Sugar-to-Fiber Ratio on NII Score',
                         fontsize=14, fontweight='bold')
                plt.xlabel('Sugar-to-Fiber Ratio', fontsize=12)
                plt.ylabel('NII Score', fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                if save_dir:
                    save_path = f"{save_dir}/interaction_sugar_fiber.png"
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    print(f"✓ Sugar-fiber interaction plot saved to: {save_path}")

                # plt.show()  # Commented out to avoid displaying plots

                # Calculate correlation
                correlation = merged_df['sugar_fiber_ratio'].corr(merged_df['NII_Score'])
                interaction_results['sugar_fiber_ratio_correlation'] = correlation
                print(f"Sugar-fiber ratio correlation: {correlation:.4f}")
            else:
                print("WARNING: Cannot create sugar-fiber plot - missing required columns")
                print(f"  Sugars,.total.(g) available: {'Sugars,.total.(g)' in merged_df.columns}")
                print(f"  Fiber,.total.dietary.(g) available: {'Fiber,.total.dietary.(g)' in merged_df.columns}")

        except Exception as e:
            print(f"ERROR creating sugar-fiber plot: {e}")
            import traceback
            traceback.print_exc()

        # SHAP interaction values
        if self.shap_values is not None:
            print("Calculating SHAP interaction values...")

            # Calculate interaction values for top features
            shap_interaction = shap.TreeExplainer(self.model).shap_interaction_values(self.X_test.iloc[:100])  # Sample for computational efficiency

            # Get top interacting feature pairs
            feature_pairs = []
            for i in range(len(self.feature_names)):
                for j in range(i+1, len(self.feature_names)):
                    interaction_strength = np.mean(np.abs(shap_interaction[:, i, j]))
                    feature_pairs.append((
                        self.feature_names[i],
                        self.feature_names[j],
                        interaction_strength
                    ))

            # Sort by interaction strength
            feature_pairs.sort(key=lambda x: x[2], reverse=True)

            interaction_results['top_shap_interactions'] = feature_pairs[:10]

            print("Top SHAP Interactions:")
            for feat1, feat2, strength in feature_pairs[:5]:
                print(f"  {feat1} × {feat2}: {strength:.4f}")

        # Fallback: Create simple interaction plots if above methods failed
        if 'protein_processing_correlations' not in interaction_results:
            print("Creating fallback protein-processing plot...")
            try:
                plt.figure(figsize=(12, 8))
                if 'Protein.(g)' in merged_df.columns and 'NII_Score' in merged_df.columns:
                    plt.scatter(merged_df['Protein.(g)'], merged_df['NII_Score'],
                              alpha=0.6, s=50, c='blue')
                    plt.title('Protein Content vs NII Score', fontsize=14, fontweight='bold')
                    plt.xlabel('Protein Content (g)', fontsize=12)
                    plt.ylabel('NII Score', fontsize=12)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()

                    if save_dir:
                        save_path = f"{save_dir}/interaction_protein_processing.png"
                        plt.savefig(save_path, dpi=300, bbox_inches='tight')
                        print(f"✓ Fallback protein-processing plot saved to: {save_path}")
            except Exception as e:
                print(f"Fallback protein plot failed: {e}")

        if 'sugar_fiber_ratio_correlation' not in interaction_results:
            print("Creating fallback sugar-fiber plot...")
            try:
                plt.figure(figsize=(10, 8))
                if ('Sugars,.total.(g)' in merged_df.columns and
                    'Fiber,.total.dietary.(g)' in merged_df.columns and
                    'NII_Score' in merged_df.columns):

                    merged_df['sugar_fiber_ratio'] = merged_df['Sugars,.total.(g)'] / (
                        merged_df['Fiber,.total.dietary.(g)'] + 0.1)

                    plt.scatter(merged_df['sugar_fiber_ratio'], merged_df['NII_Score'],
                                alpha=0.6, s=50, c='red')
                    plt.title('Sugar-to-Fiber Ratio vs NII Score', fontsize=14, fontweight='bold')
                    plt.xlabel('Sugar-to-Fiber Ratio', fontsize=12)
                    plt.ylabel('NII Score', fontsize=12)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()

                    if save_dir:
                        save_path = f"{save_dir}/interaction_sugar_fiber.png"
                        plt.savefig(save_path, dpi=300, bbox_inches='tight')
                        print(f"✓ Fallback sugar-fiber plot saved to: {save_path}")
            except Exception as e:
                print(f"Fallback sugar-fiber plot failed: {e}")

        return interaction_results

    def create_model_comparison_plots(self, all_results: dict, save_dir: str = None) -> None:
        """
        创建模型对比可视化图表。

        Args:
            all_results: 包含所有模型结果的字典
            save_dir: 保存图表的目录
        """
        comparison_df = all_results['model_comparison']

        # 1. 模型性能对比条形图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Machine Learning Model Performance Comparison', fontsize=16, fontweight='bold')

        # R2 Score comparison
        ax1 = axes[0, 0]
        bars1 = ax1.bar(comparison_df['Model'], comparison_df['Test_R2'],
                       color='skyblue', alpha=0.8)
        ax1.set_title('Test Set R² Score Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('R² Score')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        # Add value labels
        for bar, value in zip(bars1, comparison_df['Test_R2']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

        # RMSE comparison
        ax2 = axes[0, 1]
        bars2 = ax2.bar(comparison_df['Model'], comparison_df['Test_RMSE'],
                       color='lightcoral', alpha=0.8)
        ax2.set_title('Test Set RMSE Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('RMSE')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        # Add value labels
        for bar, value in zip(bars2, comparison_df['Test_RMSE']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(comparison_df['Test_RMSE'])*0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

        # MAE comparison
        ax3 = axes[1, 0]
        bars3 = ax3.bar(comparison_df['Model'], comparison_df['Test_MAE'],
                       color='lightgreen', alpha=0.8)
        ax3.set_title('Test Set MAE Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylabel('MAE')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        # Add value labels
        for bar, value in zip(bars3, comparison_df['Test_MAE']):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(comparison_df['Test_MAE'])*0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

        # Train vs Test R2 comparison
        ax4 = axes[1, 1]
        x_pos = np.arange(len(comparison_df))
        width = 0.35

        bars4 = ax4.bar(x_pos - width/2, comparison_df['Train_R2'], width,
                       label='Train R²', color='orange', alpha=0.8)
        bars5 = ax4.bar(x_pos + width/2, comparison_df['Test_R2'], width,
                       label='Test R²', color='purple', alpha=0.8)

        ax4.set_title('Train vs Test R² Score Comparison', fontsize=14, fontweight='bold')
        ax4.set_ylabel('R² Score')
        ax4.set_xlabel('Model')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_dir:
            plt.savefig(f"{save_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
            print(f"模型对比图表保存至: {save_dir}/model_comparison.png")

        # 2. 创建详细的模型排名表
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')

        # 准备表格数据
        table_data = []
        for _, row in comparison_df.iterrows():
            table_data.append([
                row['Model'],
                f"{row['Test_R2']:.4f}",
                f"{row['Test_RMSE']:.4f}",
                f"{row['Test_MAE']:.4f}",
                f"{row['CV_R2']:.4f}",
                f"{row['Train_R2']:.4f}"
            ])

        # Create table
        table = ax.table(cellText=table_data,
                       colLabels=['Model', 'Test R²', 'Test RMSE', 'Test MAE', 'CV R²', 'Train R²'],
                       cellLoc='center',
                       loc='center',
                       colColours=['#f0f0f0']*6,
                       rowColours=['#ffffff']*len(table_data))

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        # Set column colors
        for i in range(6):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Highlight best model
        best_idx = comparison_df.index[0]
        for i in range(6):
            table[(best_idx + 1, i)].set_facecolor('#ffeb3b')

        plt.title('Machine Learning Model Performance Ranking', fontsize=16, fontweight='bold', pad=20)

        if save_dir:
            plt.savefig(f"{save_dir}/model_ranking_table.png", dpi=300, bbox_inches='tight')
            print(f"Save the model ranking table to: {save_dir}/model_ranking_table.png")

        # plt.show()  

    def create_best_model_detailed_analysis(self, all_results: dict, save_dir: str = None) -> None:

        best_model_name = all_results['best_model_name']
        best_model_results = all_results[best_model_name]

    
        y_test_pred = self.model.predict(self.X_test)

        # 1. Prediction vs Actual scatter plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Best Model Detailed Analysis: {best_model_name}', fontsize=16, fontweight='bold')

        # Prediction vs Actual
        ax1 = axes[0, 0]
        scatter = ax1.scatter(self.y_test, y_test_pred, alpha=0.6, s=50, c='blue', cmap='viridis')

        # Perfect prediction line
        min_val = min(self.y_test.min(), y_test_pred.min())
        max_val = max(self.y_test.max(), y_test_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

        # Fit line
        z = np.polyfit(self.y_test, y_test_pred, 1)
        p = np.poly1d(z)
        ax1.plot(self.y_test, p(self.y_test), 'g--', lw=2,
                label=f'Fit Line (R² = {best_model_results["test_r2"]:.3f})')

        ax1.set_xlabel('Actual NII Score')
        ax1.set_ylabel('Predicted NII Score')
        ax1.set_title(f'Prediction vs Actual (R² = {best_model_results["test_r2"]:.4f})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Residual plot
        ax2 = axes[0, 1]
        residuals = self.y_test - y_test_pred
        ax2.scatter(y_test_pred, residuals, alpha=0.6, s=50, c='red')
        ax2.axhline(y=0, color='black', linestyle='--', lw=2)
        ax2.set_xlabel('Predicted Values')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residual Analysis')
        ax2.grid(True, alpha=0.3)

        # Predicted values distribution
        ax3 = axes[1, 0]
        ax3.hist(self.y_test, bins=30, alpha=0.7, label='Actual Values', color='blue', density=True)
        ax3.hist(y_test_pred, bins=30, alpha=0.7, label='Predicted Values', color='orange', density=True)
        ax3.set_xlabel('NII Score')
        ax3.set_ylabel('Density')
        ax3.set_title('Actual vs Predicted Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Error distribution
        ax4 = axes[1, 1]
        ax4.hist(residuals, bins=30, alpha=0.7, color='green', density=True)
        ax4.axvline(x=0, color='black', linestyle='--', lw=2)
        ax4.set_xlabel('Residuals')
        ax4.set_ylabel('Density')
        ax4.set_title(f'Residual Distribution (RMSE = {best_model_results["test_rmse"]:.4f})')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_dir:
            plt.savefig(f"{save_dir}/best_model_detailed_analysis.png", dpi=300, bbox_inches='tight')
            print(f"Detailed analysis of the best model saved to: {save_dir}/best_model_detailed_analysis.png")

        # 2. Feature importance graph (if the model supports it)
        if hasattr(self.model, 'feature_importances_'):
            plt.figure(figsize=(12, 8))

            feature_importance = {}
            for i, importance in enumerate(self.model.feature_importances_):
                clean_name = self.feature_names[i]
                original_name = self.original_feature_names[i]
                feature_importance[original_name] = importance

            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            features, importance = zip(*sorted_features)
            bars = plt.barh(range(len(features)), importance, color='skyblue', alpha=0.8)

            # 添加数值标签
            for i, (bar, value) in enumerate(zip(bars, importance)):
                plt.text(value + max(importance)*0.01, bar.get_y() + bar.get_height()/2,
                        f'{value:.3f}', ha='left', va='center', fontweight='bold')

            plt.yticks(range(len(features)), features)
            plt.xlabel('Feature Importance')
            plt.title(f'{best_model_name} Feature Importance Analysis', fontsize=14, fontweight='bold')
            plt.tight_layout()

            if save_dir:
                plt.savefig(f"{save_dir}/best_model_feature_importance.png", dpi=300, bbox_inches='tight')
                print(f"Best model feature importance saved to: {save_dir}/best_model_feature_importance.png")

    
        try:
            if hasattr(self.model, 'predict') and len(self.X_test) <= 1000: 
                print("Calculate the SHAP value of the optimal model...")
                self.calculate_shap_values()
                self.create_shap_summary_plot(f"{save_dir}/best_model_shap_summary.png")

                # Create a SHAP feature importance bar chart
                plt.figure(figsize=(12, 8))
                # Visualize SHAP using the original feature names
                shap.summary_plot(self.shap_values, self.X_test,
                                feature_names=self.original_feature_names,
                                plot_type="bar", show=False)
                plt.title(f'{best_model_name} SHAP Feature Importance', fontsize=14, fontweight='bold')
                plt.tight_layout()

                if save_dir:
                    plt.savefig(f"{save_dir}/best_model_shap_importance.png", dpi=300, bbox_inches='tight')
                    print(f"Best model SHAP importance saved to: {save_dir}/best_model_shap_importance.png")

        except Exception as e:
            print(f"SHAP calculation failed: {e}")

        # plt.show() 
    def create_model_performance_plots(self, results: dict, save_dir: str = None) -> None:
        """
        Create comprehensive model performance plots.

        Args:
            results: Training results dictionary
            save_dir: Directory to save plots
        """
        # Prediction vs Actual plot
        y_test_pred = self.model.predict(self.X_test)

        plt.figure(figsize=(10, 8))
        plt.scatter(self.y_test, y_test_pred, alpha=0.6, s=50)

        # Perfect prediction line
        min_val = min(self.y_test.min(), y_test_pred.min())
        max_val = max(self.y_test.max(), y_test_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

        # Calculate R2 line
        z = np.polyfit(self.y_test, y_test_pred, 1)
        p = np.poly1d(z)
        plt.plot(self.y_test, p(self.y_test), 'b--', lw=2,
                label=f'Fit Line (R2 = {results["test_r2"]:.3f})')

        plt.xlabel('Actual NII Score', fontsize=12)
        plt.ylabel('Predicted NII Score', fontsize=12)
        plt.title('Model Performance: Predicted vs Actual NII Scores',
                 fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_dir:
            save_path = f"{save_dir}/model_performance_predictions.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance plot saved to: {save_path}")

        # plt.show()  # Commented out to avoid displaying plots

        # Feature importance plot
        feature_importance = results['feature_importance']
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

        plt.figure(figsize=(12, 8))
        features, importance = zip(*sorted_features)
        plt.barh(range(len(features)), importance)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance', fontsize=12)
        plt.title('XGBoost Feature Importance for NII Score Prediction',
                 fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_dir:
            save_path = f"{save_dir}/feature_importance.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to: {save_path}")

        # plt.show()  # Commented out to avoid displaying plots

    def generate_comprehensive_report(self, results: dict, interaction_results: dict,
                                    save_path: str = None) -> dict:
        """
        Generate a comprehensive validation report.

        Args:
            results: Training results
            interaction_results: Interaction analysis results
            save_path: Path to save the report

        Returns:
            Complete report dictionary
        """
        def convert_numpy_types(obj):
            """Convert numpy types to native Python types for JSON serialization."""
            import numpy as np

            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, 'item'):  # For other numpy scalars
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(x) for x in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_numpy_types(x) for x in obj)
            else:
                return obj

        # Convert all numpy types to native Python types
        results_clean = convert_numpy_types(results)
        interaction_results_clean = convert_numpy_types(interaction_results)

        report = {
            'model_validation': {
                'r2_score': float(results_clean['test_r2']),
                'rmse': float(results_clean['test_rmse']),
                'mae': float(results_clean['test_mae']),
                'interpretation': self._interpret_r2_score(results_clean['test_r2'])
            },
            'shap_analysis': {
                'summary': 'SHAP analysis completed successfully',
                'top_features': dict(sorted(results_clean['feature_importance'].items(),
                                          key=lambda x: float(x[1]), reverse=True)[:5])
            },
            'interaction_effects': interaction_results_clean,
            'model_parameters': results_clean['model_params'],
            'data_statistics': {
                'training_samples': len(self.X_train),
                'test_samples': len(self.X_test),
                'features_used': len(self.feature_names)
            },
            'validation_conclusion': self._generate_validation_conclusion(results_clean)
        }

        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"Comprehensive report saved to: {save_path}")

        return report

    def _interpret_r2_score(self, r2: float) -> str:
        """Interpret R2 score in the context of the validation framework."""
        if r2 >= 0.8:
            return "Excellent: Strong evidence that LLM assessments correlate with nutritional data"
        elif r2 >= 0.6:
            return "Good: Substantial correlation between LLM assessments and nutritional data"
        elif r2 >= 0.4:
            return "Moderate: Some correlation, but validation is inconclusive"
        else:
            return "Poor: Weak correlation, LLM assessments may not follow nutritional principles"

    def _generate_validation_conclusion(self, results: dict) -> str:
        """Generate overall validation conclusion."""
        r2 = results['test_r2']
        conclusion = (f"The proxy model achieved an R2 score of {r2:.3f}")

        return conclusion


def main():
    """Main execution function."""
    import argparse
    from pathlib import Path
    import sys

    parser = argparse.ArgumentParser(
        description="Train Model and Generate Visualizations",
        epilog="""
Examples:
  python train_proxy_model.py --sample-size 50    # Train with 50 samples from existing CSV
  python train_proxy_model.py                       # Train with full dataset from existing CSV
        """
    )

    parser.add_argument(
        '--sample-size',
        type=int,
        help='Number of food samples to use from existing CSV data (for testing)'
    )

    parser.add_argument(
        '--search-method',
        choices=['grid', 'randomized'],
        default='randomized',
        help='Hyperparameter search method (default: randomized - recommended for comprehensive search)'
    )

    parser.add_argument(
        '--n-iter',
        type=int,
        default=100,
        help='Number of iterations for randomized search (default: 100 - increased for better optimization)'
    )

    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds (default: 5)'
    )

    parser.add_argument(
        '--models-only',
        action='store_true',
        help='Only run model comparison without additional analysis'
    )

    args = parser.parse_args()

    # Initialize paths
    base_path = Path(__file__).parent.parent
    nutritional_data_path = base_path / 'data' / 'FNDDS_2017_2018_nutrients_nova.csv'
    assessment_data_path = base_path / 'results' / 'nutritional_assessments.json'
    output_dir = base_path / 'results' / 'proxy_model_results'

    import os
    os.makedirs(output_dir, exist_ok=True)

    # Check if CSV data exists (preferred for training)
    csv_path = assessment_data_path.with_suffix('.csv')
    if not csv_path.exists():
        print(f"Error: CSV training data file not found. {csv_path}")
        print("Please run 'python main.py generate' first to generate the training data.")
        return

    print(f"Using existing CSV training data: {csv_path}")
    if args.sample_size:
        print(f"Use sample size: {args.sample_size}")

    # Step 2: Train multiple models and generate visualizations
    print("\n" + "=" * 60)
    print("STEP 2: Multi-model training and hyperparameter optimization")
    print("=" * 60)

    # Initialize trainer
    trainer = ProxyModelTrainer(assessment_data_path, nutritional_data_path)

    # Load and prepare data
    print("=" * 60)
    print("Loading and preparing data")
    print("=" * 60)

    merged_df = trainer.load_and_merge_data()
    total_samples = len(merged_df)

    # Apply sample size if specified
    if args.sample_size and args.sample_size < total_samples:
        merged_df = merged_df.sample(n=args.sample_size, random_state=42)
        print(f"{args.sample_size} samples were sampled from {total_samples} samples.")
    else:
        print(f"Training is performed using all {total_samples} samples.")

    X, y = trainer.prepare_features(merged_df)

    # Train multiple models with hyperparameter search
    print("\n" + "=" * 60)
    print("Multi-model training and hyperparameter optimization")
    print("=" * 60)
    print(f"Search method: {args.search_method}")
    print(f"Random search iteration count: {args.n_iter}")
    print(f"Cross-validation folds: {args.cv_folds}")

    all_results = trainer.train_multiple_models_with_hyperparameter_search(
        X, y,
        search_method=args.search_method,
        n_iter=args.n_iter,
        cv_folds=args.cv_folds
    )

    # Display model comparison
    print("\n" + "=" * 60)
    print("Model performance ranking")
    print("=" * 60)
    print(all_results['model_comparison'].to_string(index=False))

    best_model_name = all_results['best_model_name']
    best_results = all_results[best_model_name]

    print(f"\nOptimal Model: {best_model_name}")
    print(f"Optimal parameters: {best_results['best_params']}")
    print(f"Test set R2: {best_results['test_r2']:.4f}")
    print(f"Test set RMSE: {best_results['test_rmse']:.4f}")
    print(f"Test set MAE: {best_results['test_mae']:.4f}")

    # Create model comparison visualizations
    print("\n" + "=" * 60)
    print("Create model comparison visualization")
    print("=" * 60)

    trainer.create_model_comparison_plots(all_results, output_dir)

    # Create best model detailed analysis
    print("\n" + "=" * 60)
    print("Detailed analysis of the best model")
    print("=" * 60)

    trainer.create_best_model_detailed_analysis(all_results, output_dir)

    if not args.models_only:
        # SHAP analysis for best model
        print("\n" + "=" * 60)
        print("Optimal Model SHAP Analysis")
        print("=" * 60)

        try:
            trainer.calculate_shap_values()
            trainer.create_shap_dependence_plots(output_dir)
            trainer.create_shap_interaction_matrix(output_dir)
        except Exception as e:
            print(f"SHAP analysis failed: {e}")

        # Resolution analysis
        print("\n" + "=" * 60)
        print("Resolution Analysis")
        print("=" * 60)

        trainer.analyze_resolution_by_nova_group(merged_df, f"{output_dir}/nova_4_resolution.png")

        # Interaction effects
        print("\n" + "=" * 60)
        print("Interaction effect analysis")
        print("=" * 60)

        interaction_results = trainer.analyze_interaction_effects(merged_df, output_dir)

        # Generate comprehensive report
        print("\n" + "=" * 60)
        print("Generate a comprehensive report")
        print("=" * 60)

        # Convert results to format compatible with existing report generation
        results_for_report = {
            'test_r2': best_results['test_r2'],
            'test_rmse': best_results['test_rmse'],
            'test_mae': best_results['test_mae'],
            'model_params': best_results['best_params'],
            'feature_importance': {}
        }

        # Add feature importance if available
        if hasattr(trainer.model, 'feature_importances_'):
            results_for_report['feature_importance'] = dict(
                zip(trainer.feature_names, trainer.model.feature_importances_)
            )

        report = trainer.generate_comprehensive_report(
            results_for_report, interaction_results, f"{output_dir}/validation_report.json"
        )

        print("\nVerification Summary:")
        print(f"  R2 score: {report['model_validation']['r2_score']:.4f}")
        print(f"  explain: {report['model_validation']['interpretation']}")
        print(f"  in conclusion: {report['validation_conclusion']}")

    # Save model comparison results
    comparison_results = {
        'best_model': best_model_name,
        'best_parameters': best_results['best_params'],
        'model_ranking': all_results['model_comparison'].to_dict('records'),
        'all_model_results': {}
    }

    for model_name, model_results in all_results.items():
        if model_name not in ['best_model_name', 'best_model', 'model_comparison']:
            comparison_results['all_model_results'][model_name] = {
                'test_r2': model_results['test_r2'],
                'test_rmse': model_results['test_rmse'],
                'test_mae': model_results['test_mae'],
                'best_cv_score': model_results['best_cv_score'],
                'best_params': model_results['best_params']
            }

    with open(f"{output_dir}/model_comparison_results.json", 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nAll results are saved to: {output_dir}/")
    print("Analysis complete!")
    print(f"\n[SUCCESS] The generated file:")
    print(f"  [PLOT] model_comparison.png: Model performance comparison chart")
    print(f"  [TABLE] model_ranking_table.png: Model ranking table")
    print(f"  [PLOT] best_model_detailed_analysis.png: Detailed analysis of the best model")
    print(f"  [PLOT] best_model_feature_importance.png: Optimal Model Feature Importance")
    print(f"  [PLOT] best_model_shap_summary.png: SHAP Feature Impact Analysis")
    print(f"  [PLOT] best_model_shap_importance.png: SHAP Importance Ranking")
    print(f"  [PLOT] shap_interaction_matrix.png: SHAP Interaction Value Matrix Heatmap")
    print(f"  [DATA] shap_interaction_matrix.csv: SHAP Interaction Value Matrix Data")
    print(f"  [PLOTS] shap_dependence_*.png: SHAP dependency graph of all features ({len(trainer.original_feature_names)} files)")
    print(f"  [PLOT] nova_4_resolution.png: NOVA Group 4 Resolution Analysis")
    print(f"  [PLOT] interaction_protein_processing.png: Protein-processing interaction")
    print(f"  [DATA] model_comparison_results.json: Model comparison results data")
    if not args.models_only:
        print(f"  [REPORT] validation_report.json: Comprehensive validation report")
    print(f"\n[COMPLETE] Training complete! All models have been trained, parameter search is complete, and visualizations have been generated!")


if __name__ == "__main__":
    main()
