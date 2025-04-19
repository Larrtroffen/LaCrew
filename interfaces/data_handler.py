from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import pandas as pd

class DataHandler(ABC):
    """数据处理器接口"""
    
    @abstractmethod
    def validate_data(self, data: List[Dict[str, Any]], schema: pd.DataFrame) -> List[Dict[str, Any]]:
        """验证数据"""
        pass
    
    @abstractmethod
    def merge_data(self, existing_data: pd.DataFrame, new_data: pd.DataFrame) -> pd.DataFrame:
        """合并数据"""
        pass
    
    @abstractmethod
    def export_data(self, data: pd.DataFrame, format: str = 'csv') -> bytes:
        """导出数据"""
        pass
    
    @abstractmethod
    def deduplicate(self, data: pd.DataFrame) -> pd.DataFrame:
        """去重"""
        pass 