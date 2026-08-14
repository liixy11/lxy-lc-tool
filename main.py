#-*- coding:GBK -*-
# main.py
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings  #type:ignore
from langchain_chroma import Chroma       #type:ignore
from create import create_rag
import os
import json
import uuid
import gc

SESSION_DIR = "./chat_sessions"
VECTOR_DB = None

load_dotenv()

# 将当前会话完整保存到 JSON 文件中。
def save_chat(session_id, session_title, messages):
    data = []

    for msg in messages:
        data.append(
            {
                    "role": msg.type,
                    "content": msg.content
            }
        )

    file_path = os.path.join(
        SESSION_DIR,
        session_id + ".json"
    )

    # 写入 JSON 文件（自动覆盖已存在的同名文件
    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
        {
            "title": session_title,
            "messages": data
        },
        f,
        ensure_ascii=False,
        indent=2
)

def load_chat(session_id):

    file_path = os.path.join(
        SESSION_DIR,
        session_id + ".json"
    )


    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)


    history = InMemoryChatMessageHistory()

    # 遍历历史消息，还原到内存中
    for msg in data["messages"]:

        if msg["role"] == "human":

            history.add_user_message(
                msg["content"]
            )

        else:
            history.add_ai_message(
                msg["content"]
            )


    return history

# 根据用户输入生成标题
def create_title(text):

    max_length = 20

    if len(text) > max_length:

        return text[:max_length] + "..."

    else:

        return text


#  扫描 SESSION_DIR 目录，获取所有已保存会话的摘要信息。
def get_sessions():

    sessions = []

    for filename in os.listdir(SESSION_DIR):

        if filename.endswith(".json"):

            path = os.path.join(
                SESSION_DIR,
                filename
            )

            with open(
                path,
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)


            sessions.append(
                {
                    "id": filename.replace(".json",""),
                    "title": data.get(
                        "title",
                        "未命名会话"
                    )
                }
            )

    return sessions

# 将检索器包装成函数，方便更新调用
def load_retriever(embeddings):

    global VECTOR_DB

    VECTOR_DB = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    retriever = VECTOR_DB.as_retriever(
        search_kwargs={
            "k":3
        }
    )

    return retriever

def release_chroma():

        global VECTOR_DB

        if VECTOR_DB is not None:

            try:
                # 关闭Chroma底层客户端
                VECTOR_DB._client.close()

                print("AI：已关闭知识库连接")

            except Exception as e:

                print(
                    f"关闭知识库连接失败:{e}"
                )

        VECTOR_DB = None

        gc.collect()

def main():
    os.makedirs(SESSION_DIR,exist_ok=True)

    session_id = str(uuid.uuid4())[:8]
    session_title = None;


    print(
        f"当前会话：{session_id}"
    )

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

        retriever = load_retriever(embeddings)

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
            3. 用户可以通过 /resume 命令选择以前的聊天继续交流

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

        # 更新知识库
        if user_text == "/rag update":

            print("AI：正在更新知识库，请稍候...")

            #释放检索器
            retriever = None

            release_chroma()

            success = create_rag()

            if success:

                retriever = load_retriever(
                    embeddings
                )

                print(
                    "AI：知识库更新完成"
                )
            else:

                print(
                    "AI：知识库更新失败，请关闭占用知识库的程序后重试"
                )

            continue

         #开启RAG
        if user_text == "/rag on":

            # 检查检索器是否存在
            if retriever is None:
                print("AI：当前没有知识库，请先运行 create_db.py")
                continue
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


        if user_text == "/resume":

            sessions = get_sessions()

            if not sessions:
                print("AI：暂无历史会话")
                continue

            print("历史记录：")

            for i, session in enumerate(sessions):

                print(
                    f"{i+1}. {session['title']}"
                )

            print("\n")
            choice = input("选择编号：")

            index = int(choice) - 1


            if index < 0 or index >= len(sessions):
                print("AI：编号错误")
                continue

            target_session = sessions[index]

            print(f"选择：{target_session['title']}")

            #切换会话前先保存一次当前会话
            save_chat(
                session_id,
                session_title,
                chat_history.messages
            )

            session_id = target_session["id"]

            chat_history = load_chat(session_id)

            session_title = target_session["title"]

            print(f"AI：正在恢复 {session_title}")
            print("\n")

            continue
 
        if session_title is None:
            session_title = create_title(
                user_text
            )


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

        save_chat(session_id,session_title,chat_history.messages)

        print("\n")
        


if __name__ == "__main__" :
    main()
