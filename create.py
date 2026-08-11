# -*- coding=GBK -*-
from dotenv import load_dotenv
from langchain_chroma import Chroma  # type: ignore
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  #type: ignore
import os


load_dotenv()

def main():
    loader = TextLoader("knowledges/test.txt",encoding = "utf-8") # 读取文档
    documents = loader.load()
    

    splitter = RecursiveCharacterTextSplitter(              #文档切片
        chunk_size=300,
        chunk_overlap=50
    )

    docs = splitter.split_documents(
        documents
    )

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