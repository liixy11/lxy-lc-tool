# -*- coding=GBK -*-
from dotenv import load_dotenv
from langchain_chroma import Chroma  # type: ignore
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  #type: ignore
import os
import shutil


load_dotenv()

def main():
    def load_documents(path):
        documents = []     # 用于存储所有文档对象

        for filename in os.listdir(path):
            filepath = os.path.join(path,filename)

            if filename.endswith(".txt"):
                loader = TextLoader(filepath,encoding="utf-8")

            elif filename.endswith(".pdf"):
                loader = PyPDFLoader(filepath)

            elif filename.endswith(".docx"):
                loader = Docx2txtLoader(filepath)

            elif filename.endswith(".md"):
                loader = TextLoader(filepath,encoding="utf-8")

            else:
                continue

            # 加载当前文件的所有文档片段，并添加到总列表中
            documents.extend(loader.load())

        return documents

    # 调用加载函数，从 "./knowledges" 目录读取所有文档
    documents = load_documents("./knowledges")
    
    # 创建文本分割器，用于将长文档切分成更小的块
    splitter = RecursiveCharacterTextSplitter(              
        chunk_size=500,         # 每块最大字符数
        chunk_overlap=100       # 相邻块之间重叠的字符数，避免信息丢失
    )

     # 对加载的所有文档进行切分
    docs = splitter.split_documents(documents)

    # 如果本地已存在旧的Chroma数据库目录，则删除它（确保重新创建全新数据库）
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")

    embeddings = HuggingFaceEmbeddings(                   #配置Embedding模型
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
)

    db = Chroma.from_documents(                              #创建Chroma数据库
        documents=docs,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    print("知识库创建完成")

if __name__ == "__main__":
    main()