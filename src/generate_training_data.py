import pandas as pd
import json
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
import requests
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from openai import OpenAI
import asyncio
import aiohttp
import ssl
from datetime import datetime
import os

class PathManager:
    """管理项目文件夹结构和路径"""

    def __init__(self, base_path: str = None):
        """
        初始化路径管理器

        Args:
            base_path: 项目根路径，默认为当前工作目录
        """
        if base_path is None:
            base_path = os.getcwd()

        self.base_path = Path(base_path)

        # 定义文件夹结构
        self.folders = {
            'data': self.base_path / 'data',
            'raw_data': self.base_path / 'data',
            'processed_data': self.base_path / 'results',
            'logs': self.base_path / 'logs',
            'docs': self.base_path / 'docs',
            'models': self.base_path / 'models',
            'src': self.base_path / 'src'
        }

        # 创建所有必要的文件夹
        self._create_folders()

    def _create_folders(self):
        """创建所有必要的文件夹"""
        for folder_name, folder_path in self.folders.items():
            folder_path.mkdir(parents=True, exist_ok=True)

    def get_path(self, folder_name: str, filename: str = None) -> Path:
        """
        获取指定文件夹的路径

        Args:
            folder_name: 文件夹名称 ('data', 'images', 'logs', 'processed_data')
            filename: 文件名（可选）

        Returns:
            完整路径
        """
        if folder_name not in self.folders:
            raise ValueError(f"未知的文件夹名称: {folder_name}")

        path = self.folders[folder_name]
        if filename:
            path = path / filename

        return path

# 初始化路径管理器
path_manager = PathManager()

# Configure logging
log_file = path_manager.get_path('logs', 'training_data_generation.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(log_file), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NutritionalAssessmentGenerator:
    """
    Generate nutritional assessments using LLM for proxy model training.
    """

    def __init__(self, api_key: str, max_retries: int = 3, delay: float = 1.0, concurrency_limit: int = 10):
        """
        Initialize the assessment generator.

        Args:
            api_key: OpenAI API key
            max_retries: Maximum number of retries for API calls
            delay: Delay between API calls in seconds
            concurrency_limit: Maximum number of concurrent API calls (default: 10)
        """
        self.api_key = api_key
        self.max_retries = max_retries
        self.delay = delay
        self.concurrency_limit = concurrency_limit
        self.client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

        # Few-shot examples for better LLM performance
        self.few_shot_examples = [
            {
                "food_name": "Carbonated water, cola flavor",
                "ingredients": "Carbonated water, high fructose corn syrup, caramel color, phosphoric acid, natural flavors",
                "processing": "Ultra-processed beverage formulation",
                "expected_score_range": "0-20",
                "reasoning": "High sugar content as first ingredient, artificial additives, no nutritional value"
            },
            {
                "food_name": "Whole grain bread",
                "ingredients": "Whole wheat flour, water, yeast, salt, honey",
                "processing": "Traditional bread baking",
                "expected_score_range": "70-85",
                "reasoning": "Whole grain ingredients, minimal processing, fiber content"
            },
            {
                "food_name": "Grilled chicken breast",
                "ingredients": "Chicken breast, salt, black pepper",
                "processing": "Grilling - minimal processing",
                "expected_score_range": "85-95",
                "reasoning": "Lean protein source, minimal additives, whole food"
            }
        ]

    def construct_prompt(self, food_name: str, ingredients: str, processing: str) -> str:
        """
        Construct a comprehensive prompt for LLM nutritional assessment.

        Args:
            food_name: Name of the food item
            ingredients: Ingredients list
            processing: Processing method description

        Returns:
            Formatted prompt string
        """
        # Create few-shot context
        few_shot_context = ""
        for i, example in enumerate(self.few_shot_examples, 1):
            few_shot_context += f"""
EXAMPLE {i}:
Food Name: {example['food_name']}
Ingredients: {example['ingredients']}
Processing: {example['processing']}
Expected Score Range: {example['expected_score_range']}
Reasoning: {example['reasoning']}
"""

        prompt = f"""You are an expert Food Scientist and Nutritional Epidemiologist with deep knowledge of nutritional biochemistry, food processing technology, and public health nutrition.

OBJECTIVE:
Evaluate the "Nutritional Integrity Index" (NII) for the given food item on a scale of 0.0-100.0, where:
- 0-20: Critical Risk (e.g., Soda, candy)
- 21-40: Poor (e.g., Sugary cereals, processed snacks)
- 41-60: Moderate (e.g., White bread, processed cheese)
- 61-80: Good (e.g., Whole grain products, lean meats)
- 81-100: Excellent (e.g., Fresh fruits, vegetables, nuts)

ASSESSMENT CRITERIA:
1. Matrix Effect: Heavily penalize foods where the natural food matrix is destroyed (e.g., "isolate", "extract", "refined")
2. Ingredient Hierarchy: First 3 ingredients determine the base score. Sugar/oil/fat in top 3 = significant penalty
3. Additive Semantics: Distinguish between safe additives (e.g., Ascorbic acid, tocopherols) and ultra-processing markers (e.g., artificial colors, emulsifiers, flavor enhancers)
4. Nutritional Compensation: Recognize protective nutrients (e.g., "whole beans" implies fiber, "leafy greens" implies vitamins)
5. Processing Impact: Consider how processing affects nutritional quality and bioavailability

{few_shot_context}

CURRENT ASSESSMENT:
Food Name: {food_name}
Ingredients: {ingredients}
Processing: {processing}

Please provide a detailed nutritional assessment following this exact JSON format:

{{
  "extracted_features": {{
    "is_ultra_processed": boolean,
    "primary_macronutrient_source": string,
    "negative_additives_count": int,
    "positive_whole_food_markers": list[string],
    "sugar_position_in_ingredients": int,
    "has_artificial_additives": boolean,
    "matrix_destruction_indicators": list[string]
  }},
  "reasoning_chain": "Step-by-step analysis of how you arrived at the score, considering ingredients hierarchy, processing impact, and nutritional compensation...",
  "NII_Score": float
}}

IMPORTANT:
- Base your assessment ONLY on the provided ingredients and processing information
- Consider both positive and negative nutritional aspects
- Provide a scientifically justified score
- Ensure the JSON is valid and complete
"""

        return prompt

    async def call_llm_api_async(self, prompt: str, session: aiohttp.ClientSession) -> Optional[Dict]:
        """
        Asynchronously call LLM API for nutritional assessment.

        Args:
            prompt: The assessment prompt
            session: aiohttp ClientSession for making requests

        Returns:
            LLM response as dictionary or None if failed
        """
        for attempt in range(self.max_retries):
            try:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }

                payload = {
                    "model": "openai/gpt-5.1",
                    "messages": [
                        {"role": "system", "content": "You are a precise food science expert who always responds in valid JSON format."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 1000
                }

                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    ssl=ssl.create_default_context()
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data["choices"][0]["message"]["content"]

                        # Parse JSON response
                        try:
                            assessment = json.loads(content)
                            return assessment
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse JSON response: {e}")
                            logger.error(f"Raw content: {content}")
                            if attempt < self.max_retries - 1:
                                await asyncio.sleep(self.delay * (attempt + 1))
                                continue
                            return None
                    else:
                        error_text = await response.text()
                        logger.error(f"HTTP error {response.status}: {error_text}")

            except Exception as e:
                logger.error(f"API call failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.delay * (attempt + 1) * 2)  # Exponential backoff
                    continue
                return None

        return None

    def call_llm_api(self, prompt: str) -> Optional[Dict]:
        """
        Synchronous wrapper for LLM API call (kept for compatibility).

        Args:
            prompt: The assessment prompt

        Returns:
            LLM response as dictionary or None if failed
        """
        async def wrapper():
            async with aiohttp.ClientSession() as session:
                return await self.call_llm_api_async(prompt, session)
        return asyncio.run(wrapper())

    async def assess_single_food_async(self, row: pd.Series, session: aiohttp.ClientSession) -> Optional[Dict]:
        """
        Asynchronously assess a single food item.

        Args:
            row: Pandas Series containing food information
            session: aiohttp ClientSession for making requests

        Returns:
            Assessment dictionary or None if failed
        """
        try:
            # Extract relevant information
            food_name = row.get('Main.food.description', '')
            ingredients = row.get('Combined.ingredients', '')
            processing = row.get('Processing.Method', '')

            # Skip if essential information is missing
            if not food_name or not ingredients:
                logger.warning(f"Missing essential information for {row.name}")
                return None

            # Construct prompt and get assessment
            prompt = self.construct_prompt(food_name, ingredients, processing)
            assessment = await self.call_llm_api_async(prompt, session)

            if assessment:
                # Add metadata
                assessment['food_code'] = row.get('Food.code', '')
                assessment['food_name'] = food_name
                assessment['nova_group'] = row.get('NOVA.Group', '')
                assessment['ingredients_raw'] = ingredients
                assessment['processing_raw'] = processing

                # Add timestamp
                assessment['assessment_timestamp'] = time.time()

                return assessment
            else:
                logger.error(f"Failed to get assessment for {food_name}")
                return None

        except Exception as e:
            logger.error(f"Error assessing food item {row.name}: {e}")
            return None

    def assess_single_food(self, row: pd.Series) -> Optional[Dict]:
        """
        Assess a single food item (synchronous wrapper).

        Args:
            row: Pandas Series containing food information

        Returns:
            Assessment dictionary or None if failed
        """
        try:
            # Extract relevant information
            food_name = row.get('Main.food.description', '')
            ingredients = row.get('Combined.ingredients', '')
            processing = row.get('Processing.Method', '')

            # Skip if essential information is missing
            if not food_name or not ingredients:
                logger.warning(f"Missing essential information for {row.name}")
                return None

            # Construct prompt and get assessment
            prompt = self.construct_prompt(food_name, ingredients, processing)
            assessment = self.call_llm_api(prompt)

            if assessment:
                # Add metadata
                assessment['food_code'] = row.get('Food.code', '')
                assessment['food_name'] = food_name
                assessment['nova_group'] = row.get('NOVA.Group', '')
                assessment['ingredients_raw'] = ingredients
                assessment['processing_raw'] = processing

                # Add timestamp
                assessment['assessment_timestamp'] = time.time()

                return assessment
            else:
                logger.error(f"Failed to get assessment for {food_name}")
                return None

        except Exception as e:
            logger.error(f"Error assessing food item {row.name}: {e}")
            return None

    async def generate_batch_assessments_async(self, df: pd.DataFrame, output_path: str,
                                            batch_size: int = 100, sample_size: Optional[int] = None) -> None:
        """
        Asynchronously generate assessments for a batch of food items with parallel processing.

        Args:
            df: DataFrame containing food data
            output_path: Path to save the results
            batch_size: Number of items to process before saving
            sample_size: Number of items to sample (for testing)
        """
        # Sample data if specified
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            logger.info(f"Sampled {sample_size} items from {len(df)} total items")

        logger.info(f"Starting asynchronous assessment generation for {len(df)} food items")
        logger.info(f"Concurrency limit: {self.concurrency_limit}")

        # Initialize results storage
        all_assessments = []
        failed_items = []

        # Create aiohttp session
        connector = aiohttp.TCPConnector(limit=self.concurrency_limit, force_close=True)
        timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes timeout
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Create semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(self.concurrency_limit)

            async def process_item_with_semaphore(idx: int, row: pd.Series) -> Tuple[int, Optional[Dict], Optional[Dict]]:
                async with semaphore:
                    if idx % 10 == 0:
                        logger.info(f"Processing item {idx + 1}/{len(df)}")

                    # Assess the food item asynchronously
                    assessment = await self.assess_single_food_async(row, session)

                    if assessment:
                        return idx, assessment, None
                    else:
                        failed_item = {
                            'food_code': row.get('Food.code', ''),
                            'food_name': row.get('Main.food.description', ''),
                            'error': 'Assessment failed'
                        }
                        return idx, None, failed_item

            # Process all items in parallel
            tasks = []
            for idx, (_, row) in enumerate(df.iterrows()):
                task = asyncio.create_task(process_item_with_semaphore(idx, row))
                tasks.append(task)

            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            processed_count = 0
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Task failed with exception: {result}")
                    continue

                idx, assessment, failed_item = result

                if assessment:
                    all_assessments.append(assessment)
                if failed_item:
                    failed_items.append(failed_item)

                processed_count += 1

                # Save batch results
                if processed_count % batch_size == 0:
                    # 从完整路径中提取文件名
                    output_filename = Path(output_path).name
                    self._save_intermediate_results(all_assessments, failed_items, output_filename, processed_count)

        # Save final results
        output_filename = Path(output_path).name
        self._save_final_results(all_assessments, failed_items, output_filename)

        logger.info(f"Completed asynchronous assessment generation:")
        logger.info(f"  - Successful assessments: {len(all_assessments)}")
        logger.info(f"  - Failed items: {len(failed_items)}")
        logger.info(f"  - Success rate: {len(all_assessments) / len(df) * 100:.1f}%")

    def generate_batch_assessments(self, df: pd.DataFrame, output_path: str,
                                  batch_size: int = 100, sample_size: Optional[int] = None) -> None:
        """
        Generate assessments for a batch of food items (synchronous wrapper).

        Args:
            df: DataFrame containing food data
            output_path: Path to save the results
            batch_size: Number of items to process before saving
            sample_size: Number of items to sample (for testing)
        """
        asyncio.run(self.generate_batch_assessments_async(df, output_path, batch_size, sample_size))

    def _save_intermediate_results(self, assessments: List[Dict],
                                   failed_items: List[Dict],
                                   output_filename: str,
                                   processed_count: int) -> None:
        """Save intermediate results to prevent data loss."""
        # 生成中间结果文件名
        intermediate_filename = output_filename.replace('.json', f'_intermediate_{processed_count}.json')
        intermediate_path = path_manager.get_path('processed_data', intermediate_filename)

        results = {
            'assessments': assessments,
            'failed_items': failed_items,
            'processed_count': processed_count,
            'timestamp': time.time()
        }

        with open(intermediate_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"已保存中间结果: {intermediate_path}")

    def _save_final_results(self, assessments: List[Dict],
                           failed_items: List[Dict],
                           output_filename: str) -> None:
        """Save final results."""
        final_path = path_manager.get_path('processed_data', output_filename)

        results = {
            'metadata': {
                'total_assessments': len(assessments),
                'total_failed': len(failed_items),
                'success_rate': len(assessments) / (len(assessments) + len(failed_items)) if (len(assessments) + len(failed_items)) > 0 else 0,
                'generation_timestamp': time.time(),
                'model_used': 'openai/gpt-5.1',
                'prompt_version': '1.0',
                'concurrency_limit': self.concurrency_limit
            },
            'assessments': assessments,
            'failed_items': failed_items
        }

        with open(final_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"已保存最终结果: {final_path}")

    def save_validation_results(self, validation_stats: Dict, output_filename: str) -> None:
        """Save validation results."""
        validation_filename = output_filename.replace('.json', '_validation.json')
        validation_path = path_manager.get_path('processed_data', validation_filename)

        with open(validation_path, 'w', encoding='utf-8') as f:
            json.dump(validation_stats, f, indent=2, ensure_ascii=False)

        logger.info(f"已保存验证结果: {validation_path}")

    def validate_assessments(self, assessments: List[Dict]) -> Dict:
        """
        Validate the generated assessments.

        Args:
            assessments: List of assessment dictionaries

        Returns:
            Validation statistics
        """
        if not assessments:
            return {"error": "No assessments to validate"}

        # Extract scores
        scores = [a.get('NII_Score', 0) for a in assessments]

        # Calculate statistics
        stats = {
            'total_assessments': len(assessments),
            'score_mean': np.mean(scores),
            'score_std': np.std(scores),
            'score_min': np.min(scores),
            'score_max': np.max(scores),
            'score_median': np.median(scores),
            'score_q25': np.percentile(scores, 25),
            'score_q75': np.percentile(scores, 75)
        }

        # Check score distribution
        score_ranges = {
            '0-20': sum(1 for s in scores if 0 <= s <= 20),
            '21-40': sum(1 for s in scores if 21 <= s <= 40),
            '41-60': sum(1 for s in scores if 41 <= s <= 60),
            '61-80': sum(1 for s in scores if 61 <= s <= 80),
            '81-100': sum(1 for s in scores if 81 <= s <= 100)
        }

        stats['score_distribution'] = score_ranges

        # Validate required fields
        required_fields = ['NII_Score', 'extracted_features', 'reasoning_chain']
        field_completeness = {}

        for field in required_fields:
            complete_count = sum(1 for a in assessments if field in a and a[field] is not None)
            field_completeness[field] = {
                'complete': complete_count,
                'total': len(assessments),
                'completeness_rate': complete_count / len(assessments)
            }

        stats['field_completeness'] = field_completeness

        return stats


def main():
    """Main execution function."""
    # Load configuration
    try:
        from config import api_key
    except ImportError:
        logger.error("Please create config.py with your OpenAI API key")
        return

    # Initialize paths
    data_path = path_manager.get_path('raw_data', '副本FNDDS_2017_2018_NOVA_v3_nutrients.csv')
    output_path = path_manager.get_path('processed_data', 'nutritional_assessments.json')

    # Check if data file exists
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return

    # Load data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} food items")

    # Initialize generator with concurrency limit
    generator = NutritionalAssessmentGenerator(api_key=api_key, delay=0.5, concurrency_limit=10)

    # Generate assessments without sample size limit for production use
    generator.generate_batch_assessments(
        df=df,
        output_path=output_path,
        batch_size=100  # Increased batch size for efficiency
        # Remove sample_size parameter to process full dataset
    )

    # Load and validate results
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            results = json.load(f)

        assessments = results['assessments']
        validation_stats = generator.validate_assessments(assessments)

        logger.info("Validation Results:")
        for key, value in validation_stats.items():
            if isinstance(value, dict):
                logger.info(f"  {key}:")
                for sub_key, sub_value in value.items():
                    logger.info(f"    {sub_key}: {sub_value}")
            else:
                logger.info(f"  {key}: {value}")

        # Save validation results
        validation_filename = Path(output_path).stem + '_validation.json'
        validation_path = path_manager.get_path('processed_data', validation_filename)
        with open(validation_path, 'w', encoding='utf-8') as f:
            json.dump(validation_stats, f, indent=2)

        logger.info(f"Validation results saved to: {validation_path}")

        # Convert JSON assessments to CSV for machine learning training
        logger.info("Converting assessments to CSV format for training...")
        csv_filename = Path(output_path).stem + '.csv'
        csv_path = path_manager.get_path('processed_data', csv_filename)

        # Load nutritional data to merge with assessments
        logger.info("Loading nutritional data for CSV generation...")
        nutritional_df = pd.read_csv(data_path)

        # Define the 12 nutritional features used in training
        nutritional_features = [
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

        # Extract features from assessments for CSV - only include training-relevant data
        csv_data = []
        for assessment in assessments:
            food_code = assessment.get('food_code', '')

            # Find matching nutritional data
            nutrition_row = nutritional_df[nutritional_df['Food.code'] == food_code]

            if not nutrition_row.empty:
                # Create row with only the features needed for training
                row = {
                    'food_code': food_code,
                    'food_name': assessment.get('food_name', ''),
                    'nova_group': assessment.get('nova_group', ''),
                    'NII_Score': assessment.get('NII_Score', 0)  # Target variable
                }

                # Add only the 12 nutritional features used in training
                for feature in nutritional_features:
                    if feature in nutrition_row.columns:
                        row[feature] = nutrition_row[feature].iloc[0]
                    else:
                        row[feature] = 0  # Default value if missing

                csv_data.append(row)
            else:
                logger.warning(f"No nutritional data found for food code: {food_code}")

        logger.info(f"Created CSV with {len(csv_data)} rows and {len(nutritional_features) + 4} columns")

        # Create DataFrame and save to CSV
        csv_df = pd.DataFrame(csv_data)
        csv_df.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info(f"CSV data saved to: {csv_path}")
        logger.info(f"Generated {len(csv_df)} training samples with {len(csv_df.columns)} features")

    except Exception as e:
        logger.error(f"Error validating results: {e}")


if __name__ == "__main__":
    main()