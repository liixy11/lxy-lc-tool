#-*- coding:GBK -*-
# main.py
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
    # 初始化大语言模型（LLM），使用ChatOpenAI接口
    # 从环境变量中读取模型名称、API密钥和基础URL
    llm = ChatOpenAI(       
        model=os.getenv("MODEL_NAME"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE_URL"),
    )   

    #配置Embedding模型
    embeddings = HuggingFaceEmbeddings(                   
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    # 初始化检索器
    retriever = None

    # 检查本地是否存在Chroma向量数据库目录
    if os.path.exists("./chroma_db"):

        db = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )

        # 创建检索器，设置检索类型为相似度检索，每次k个文档
        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k":3}
        )

    # 知识库不存在提示
    else:
        print("提示：未发现知识库，RAG功能不可用")


    # 聊天提示模板构建
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
    use_rag = False

    while True:
        user_text = input("用户：")

         #开启RAG
        if user_text == "/rag on":

            # 检查检索器是否存在
            if retriever is None:
                print("AI：当前没有知识库，请先运行 create_db.py")

            else:
                use_rag = True
                print("AI：知识库已开启")
                print("\n")
                continue

         #关闭RAG
        if user_text == "/rag off":
            use_rag = False
            print("AI：知识库已关闭")
            print("\n")
            continue

        # 将用户消息添加到聊天历史
        chat_history.add_user_message(user_text)  

        # 根据RAG开关状态和检索器是否存在，决定是否检索知识库
        if use_rag and retriever is not None:

            docs = retriever.invoke(user_text)

            context = "\n\n".join(
                [
                    doc.page_content
                    for doc in docs
                ]
            )
        else:
            context=""

        # 14. 调用对话链，传入完整的消息历史和上下文
        resp = chat_chain.invoke({"messages": chat_history.messages,"context":context})
        print(f"AI：{resp.content.strip()}")

        #添加AI消息到聊天记录中
        chat_history.add_ai_message(resp.content) 

        print("\n")
        


if __name__ == "__main__" :
    main()