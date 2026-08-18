# LangChain RAG CLI Chat Tool

基于 LangChain 的 RAG（检索增强生成）命令行聊天工具，支持私有知识库问答。

## 功能

- 支持 .txt / .pdf / .docx / .md 多格式文档
- 支持 /rag on|off 动态开关知识库、/rag update 热更新
- 支持多轮对话历史
- 支持会话保存与恢复（/resume）
- 流式输出（AI 逐字显示回答）
- MMR 检索算法（召回更精准）
- 离线模式（模型缓存后无需联网）
- 使用 BAAI/bge-m3 中文嵌入模型

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/liixy11/lxy-lc-tool.git
cd lxy-lc-tool
```

### 2. 创建虚拟环境

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

创建 `.env` 文件：

```
OPENAI_API_KEY=***
OPENAI_API_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o
```

支持任何兼容 OpenAI API 格式的服务。

### 5. 准备知识库

将 .txt / .pdf / .docx / .md 文档放入 `knowledges/` 目录（已内置考勤管理制度示例）。

### 6. 创建向量数据库

```bash
python create.py
```

首次运行会下载 BAAI/bge-m3 嵌入模型（约 2GB）。

### 7. 开始对话

```bash
python main.py
```

输入问题即可获得基于知识库的回答，Ctrl+C 退出。

## 内置命令

| 命令 | 说明 |
|------|------|
| /rag on | 开启知识库检索（需先运行 create.py） |
| /rag off | 关闭知识库检索 |
| /rag update | 热更新知识库（重新扫描 knowledges/ 重建向量库） |
| /resume | 列出历史会话并选择恢复 |

会话记录自动保存在本地 `chat_sessions/` 目录（已加入 .gitignore，不会上传）。

## 自定义知识库

1. 将文档放入 `knowledges/` 目录（支持 .txt / .pdf / .docx / .md）
2. `python create.py` 重建向量库
3. `python main.py` 开始问答

## 项目结构

```
.
├── main.py             交互式 RAG 聊天入口
├── create.py           从 knowledges/ 创建向量数据库
├── requirements.txt    Python 依赖
├── .env                环境变量（需自行创建）
├── knowledges/         知识库文档
├── chroma_db/          向量数据库（运行 create.py 后生成）
└── chat_sessions/      会话记录（运行 main.py 后生成）
```
