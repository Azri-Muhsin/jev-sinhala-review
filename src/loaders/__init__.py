"""
src/loaders/__init__.py — Dataset loaders for all probe tasks.
"""

from src.loaders.base import BaseLoader, DatasetRecord
from src.loaders.loader_a_sentiment import SentimentLoader
from src.loaders.loader_b_sold import SOLDLoader
from src.loaders.loader_c_nsina import NSINACategoriesLoader, NSINAMediaLoader
from src.loaders.loader_d_sinhalammlu import SinhalaMMLULoader
from src.loaders.loader_e_salangabhava import SalAngaBhavaLoader
from src.loaders.loader_f_cmcs import CMCSLoader

__all__ = [
    "BaseLoader",
    "DatasetRecord",
    "SentimentLoader",
    "SOLDLoader",
    "NSINACategoriesLoader",
    "NSINAMediaLoader",
    "SinhalaMMLULoader",
    "SalAngaBhavaLoader",
    "CMCSLoader",
]
