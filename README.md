<<<<<<< HEAD
# IntelliScrape Studio

基于LLM的智能Web数据爬取与结构化工具

## 项目简介

IntelliScrape Studio 是一个基于Streamlit的Web应用程序，允许用户通过图形化界面定义结构化数据表格，并利用大型语言模型（LLM）驱动的自主Web浏览和信息提取能力，根据用户定义的表格结构自动收集、整理和填充数据。

## 主要功能

- 图形化界面定义数据结构
- LLM驱动的智能Web爬取
- 自动数据提取与结构化
- 实时数据预览与导出
- 支持多种数据格式（CSV、Excel）

## 环境要求

- Python 3.8+
- 现代浏览器（Chrome、Firefox、Edge等）

## 安装步骤

1. 克隆项目到本地：
```bash
git clone [项目地址]
cd IntelliScrape-Studio
```

2. 创建并激活虚拟环境：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

4. 安装Playwright浏览器：
```bash
playwright install
```

5. 配置环境变量：
创建 `.env` 文件并添加以下内容：
```
OPENAI_API_KEY=your_api_key_here
```

## 运行应用

```bash
streamlit run app.py
```

## 使用说明

1. 在浏览器中打开应用（默认地址：http://localhost:8501）
2. 描述您想要收集的数据类型或目标
3. 使用界面定义数据结构
4. 点击"开始爬取"按钮
5. 等待数据收集完成
6. 导出数据到CSV或Excel文件

## 注意事项

- 请确保您有足够的API调用额度
- 建议在使用前仔细阅读数据采集目标网站的使用条款
- 合理设置爬取深度和频率，避免对目标网站造成过大压力

## 许可证

MIT License 
=======
# LaCrew
An AI-powered internet crew for structure data.

Coming soon, Simple thought is build a ai-guided crew for collect multimodal data, then translate them to structure data.
>>>>>>> bb5fe5d3b4684832f041e308a20a8ac3e5b9fd57
