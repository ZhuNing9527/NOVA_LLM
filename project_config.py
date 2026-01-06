"""
NOVA LLM 项目配置文件
包含路径管理、模型配置等
"""

import os
from pathlib import Path

class ProjectConfig:
    """项目配置类"""

    def __init__(self, base_path: str = None):
        if base_path is None:
            base_path = os.getcwd()

        self.base_path = Path(base_path)

        # 文件夹路径
        self.data_dir = self.base_path / 'data'
        self.src_dir = self.base_path / 'src'
        self.docs_dir = self.base_path / 'docs'
        self.logs_dir = self.base_path / 'logs'
        self.results_dir = self.base_path / 'results'
        self.models_dir = self.base_path / 'models'

        # 数据文件路径
        self.raw_data_file = self.data_dir / '副本FNDDS_2017_2018_NOVA_v3_nutrients.csv'
        self.validation_report = self.data_dir / 'data_validation_report.json'

        # 结果文件路径
        self.nutritional_assessments = self.results_dir / 'nutritional_assessments.json'
        self.validation_results = self.results_dir / 'nutritional_assessments_validation.json'

        # 日志文件路径
        self.training_log = self.logs_dir / 'training_data_generation.log'

        # 模型配置
        self.api_base_url = "https://openrouter.ai/api/v1"
        self.default_model = "openai/gpt-5.1"
        self.concurrency_limit = 10

        # 创建必要的目录
        self._create_directories()

    def _create_directories(self):
        """创建必要的目录"""
        directories = [
            self.data_dir, self.src_dir, self.docs_dir,
            self.logs_dir, self.results_dir, self.models_dir
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

# 全局配置实例
config = ProjectConfig()