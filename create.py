# -*- coding: GBK -*-
# create.py
from dotenv import load_dotenv
from langchain_chroma import Chroma  # type: ignore
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
import os
import shutil

load_dotenv()

def create_rag():
    def load_documents(path):
        documents = []
        for filename in os.listdir(path):
            filepath = os.path.join(path, filename)
            if filename.endswith(".txt"):
                loader = TextLoader(filepath, encoding="utf-8")
            elif filename.endswith(".pdf"):
                loader = PyPDFLoader(filepath)
            elif filename.endswith(".docx"):
                loader = Docx2txtLoader(filepath)
            elif filename.endswith(".md"):
                loader = TextLoader(filepath, encoding="utf-8")
            else:
                continue
            documents.extend(loader.load())
        return documents

    documents = load_documents("./knowledges")

    # 调整分块大小和重叠，使语义更完整
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    docs = splitter.split_documents(documents)

    # 删除旧数据库（增加异常处理）
    if os.path.exists("./chroma_db"):
        try:
            shutil.rmtree("./chroma_db")
        except PermissionError:
            print("知识库正在使用，请稍后重试")
            return False
        except Exception as e:
            print(f"删除旧知识库失败: {e}")
            return False

    # 离线加载本地缓存模型，避免连接 huggingface.co 超时
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    print("知识库创建/更新完成")
    return True

if __name__ == "__main__":
    create_rag()