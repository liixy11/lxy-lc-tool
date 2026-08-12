# LangChain RAG CLI Chat Tool

基于 LangChain 的 RAG（检索增强生成）命令行聊天工具，支持私有知识库问答。

## 功能

- 支持 .txt / .pdf / .docx / .md 多格式文档
- 支持 /rag on|off 动态开关知识库
- 支持多轮对话历史
- 使用 BAAI/bge-m3 中文嵌入模型

## 快速开始

### 1. 克隆仓库

git clone https://github.com/liixy11/lxy-lc-tool.git
cd lxy-lc-tool

### 2. 创建虚拟环境

python -m venv venv
# Windows
venv\Scripts\activate

### 3. 安装依赖

pip install -r requirements.txt

### 4. 配置环境变量

创建 .env 文件：

OPENAI_API_KEY=your-key
OPENAI_API_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o

支持任何兼容 OpenAI API 格式的服务。

### 5. 准备知识库

将 .txt / .pdf / .docx / .md 文档放入 knowledges/ 目录（已内置考勤管理制度示例）。

### 6. 创建向量数据库

python create.py

首次运行会下载 BAAI/bge-m3 嵌入模型（约 2GB）。

### 7. 开始对话

python main.py

输入问题即可获得基于知识库的回答，Ctrl+C 退出。

## 自定义知识库

1. 将文档放入 knowledges/ 目录（支持 .txt / .pdf / .docx / .md）
2. python create.py 重建向量库
3. python main.py 开始问答
4. 对话中输入 /rag on 开启知识库，/rag off 关闭

