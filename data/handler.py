from typing import List, Dict, Any
import pandas as pd
from interfaces.data_handler import DataHandler
from logging.logger import Logger
from data.models import DataModel, DataSchema
import re
from datetime import datetime

class DataHandlerImpl(DataHandler):
    """数据处理器实现"""
    
    def __init__(self, logger: Logger):
        self.logger = logger
        self.data_types = {
            'Text': str,
            'Number': float,
            'Date': datetime,
            'URL': str,
            'Email': str,
            'Boolean': bool
        }
    
    def validate_data(self, data: List[Dict[str, Any]], schema: pd.DataFrame) -> List[Dict[str, Any]]:
        """验证和清理数据"""
        self.logger.info("开始验证数据")
        
        validated_data = []
        
        for item in data:
            validated_item = {}
            for _, field in schema.iterrows():
                field_name = field['字段名']
                field_type = field['数据类型']
                
                if field_name in item:
                    value = item[field_name]
                    validated_value = self._validate_field(value, field_type)
                    if validated_value is not None:
                        validated_item[field_name] = validated_value
            
            if validated_item:
                validated_data.append(validated_item)
        
        self.logger.info(f"数据验证完成，有效数据: {len(validated_data)} 条")
        return validated_data
    
    def _validate_field(self, value: Any, field_type: str) -> Any:
        """验证单个字段的值"""
        if value is None:
            return None
            
        try:
            if field_type == 'Text':
                return str(value).strip()
                
            elif field_type == 'Number':
                # 移除可能存在的货币符号和千位分隔符
                cleaned = re.sub(r'[^\d.-]', '', str(value))
                return float(cleaned)
                
            elif field_type == 'Date':
                # 尝试多种日期格式
                date_formats = [
                    '%Y-%m-%d',
                    '%d/%m/%Y',
                    '%m/%d/%Y',
                    '%Y.%m.%d',
                    '%d.%m.%Y',
                    '%m.%d.%Y'
                ]
                
                for fmt in date_formats:
                    try:
                        return datetime.strptime(str(value), fmt)
                    except ValueError:
                        continue
                return None
                
            elif field_type == 'URL':
                url = str(value).strip()
                if url.startswith(('http://', 'https://')):
                    return url
                return None
                
            elif field_type == 'Email':
                email = str(value).strip()
                if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
                    return email
                return None
                
            elif field_type == 'Boolean':
                if isinstance(value, bool):
                    return value
                value_str = str(value).lower()
                if value_str in ('true', '1', 'yes', 'y'):
                    return True
                elif value_str in ('false', '0', 'no', 'n'):
                    return False
                return None
                
        except (ValueError, TypeError):
            return None
            
        return None
    
    def merge_data(self, existing_data: pd.DataFrame, new_data: pd.DataFrame) -> pd.DataFrame:
        """合并新旧数据，去除重复"""
        self.logger.info("开始合并数据")
        
        if existing_data.empty:
            return new_data
        if new_data.empty:
            return existing_data
            
        # 合并数据框
        merged = pd.concat([existing_data, new_data], ignore_index=True)
        
        # 去除完全重复的行
        merged = merged.drop_duplicates()
        
        self.logger.info(f"数据合并完成，合并后数据: {len(merged)} 条")
        return merged
    
    def export_data(self, data: pd.DataFrame, format: str = 'csv') -> bytes:
        """导出数据为指定格式"""
        self.logger.info(f"开始导出数据，格式: {format}")
        
        if format.lower() == 'csv':
            return data.to_csv(index=False).encode('utf-8')
        elif format.lower() == 'excel':
            import io
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                data.to_excel(writer, index=False)
            return output.getvalue()
        else:
            raise ValueError(f"不支持的导出格式: {format}")
    
    def deduplicate(self, data: pd.DataFrame) -> pd.DataFrame:
        """去重"""
        try:
            self.logger.info("开始数据去重")
            
            # 去除完全重复的行
            deduplicated = data.drop_duplicates()
            
            self.logger.info(f"数据去重完成，去重后数据: {len(deduplicated)} 条")
            return deduplicated
            
        except Exception as e:
            self.logger.error(f"数据去重失败: {str(e)}")
            return data 