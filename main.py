#-*- coding:GBK -*-
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings  #type:ignore
from langchain_chroma import Chroma       #type:ignore
import os


load_dotenv()

def main():
    llm = ChatOpenAI(       
        model=os.getenv("MODEL_NAME"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE_URL"),
    )   

    embeddings = HuggingFaceEmbeddings(                   #配置Embedding模型
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    db = Chroma(
        persist_directory="./chroma_db",       # 指向已存在的向量库目录
        embedding_function=embeddings          # 必须使用与创建时相同的嵌入模型
    )

    retriever = db.as_retriever(
        search_kwargs={"k":1}                  # 每次检索返回最相似的1个文档片段
    )

    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            你是一个有用的AI助手。

            回答规则：
            1. 当问到知识库中有相关内容的问题优先根据知识库回答，没被问到则不用管。
            2. 结合历史聊天理解用户问题。

            知识库内容：

            {context}

            """
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


    chat_chain = prompt | llm
    chat_history = InMemoryChatMessageHistory()

    while True:
        user_text = input("用户：")
        chat_history.add_user_message(user_text)  # 保存用户消息
        docs = retriever.invoke(user_text)
        context="\n\n".join(
            [
                doc.page_content
                for doc in docs
            ]
        )
        resp = chat_chain.invoke({"messages": chat_history.messages,"context":context})
        print(f"AI：{resp.content.strip()}")
        chat_history.add_ai_message(resp.content) 
        print("\n")


if __name__ == "__main__" :
    main()