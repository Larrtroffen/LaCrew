from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime
import re

class DataField:
    """数据字段定义"""
    
    def __init__(self, name: str, data_type: str, description: str = ""):
        self.name = name
        self.data_type = data_type
        self.description = description
    
    def validate(self, value: Any) -> Any:
        """验证字段值"""
        if value is None:
            return None
            
        try:
            if self.data_type == 'Text':
                return str(value).strip()
                
            elif self.data_type == 'Number':
                cleaned = re.sub(r'[^\d.-]', '', str(value))
                return float(cleaned)
                
            elif self.data_type == 'Date':
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
                
            elif self.data_type == 'URL':
                url = str(value).strip()
                if url.startswith(('http://', 'https://')):
                    return url
                return None
                
            elif self.data_type == 'Email':
                email = str(value).strip()
                if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
                    return email
                return None
                
            elif self.data_type == 'Boolean':
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

class DataSchema:
    """数据结构定义"""
    
    def __init__(self):
        self.fields: List[DataField] = []
    
    def add_field(self, field: DataField):
        """添加字段"""
        self.fields.append(field)
    
    def get_field(self, name: str) -> Optional[DataField]:
        """获取字段"""
        for field in self.fields:
            if field.name == name:
                return field
        return None
    
    def to_dataframe(self) -> pd.DataFrame:
        """转换为DataFrame"""
        data = {
            '字段名': [field.name for field in self.fields],
            '数据类型': [field.data_type for field in self.fields],
            '描述': [field.description for field in self.fields]
        }
        return pd.DataFrame(data)
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> 'DataSchema':
        """从DataFrame创建"""
        schema = cls()
        for _, row in df.iterrows():
            field = DataField(
                name=row['字段名'],
                data_type=row['数据类型'],
                description=row['描述']
            )
            schema.add_field(field)
        return schema

class DataModel:
    """数据模型"""
    
    def __init__(self, schema: DataSchema):
        self.schema = schema
        self.data: List[Dict[str, Any]] = []
    
    def add_record(self, record: Dict[str, Any]) -> bool:
        """添加记录"""
        validated_record = {}
        for field in self.schema.fields:
            value = record.get(field.name)
            validated_value = field.validate(value)
            if validated_value is not None:
                validated_record[field.name] = validated_value
        
        if validated_record:
            self.data.append(validated_record)
            return True
        return False
    
    def to_dataframe(self) -> pd.DataFrame:
        """转换为DataFrame"""
        return pd.DataFrame(self.data)
    
    def merge(self, other: 'DataModel') -> 'DataModel':
        """合并数据模型"""
        merged = DataModel(self.schema)
        merged.data = self.data + other.data
        return merged
    
    def deduplicate(self):
        """去重"""
        df = self.to_dataframe()
        df = df.drop_duplicates()
        self.data = df.to_dict('records')
    
    def export(self, format: str = 'csv') -> bytes:
        """导出数据"""
        df = self.to_dataframe()
        if format.lower() == 'csv':
            return df.to_csv(index=False).encode('utf-8')
        elif format.lower() == 'excel':
            import io
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            return output.getvalue()
        else:
            raise ValueError(f"不支持的导出格式: {format}") 