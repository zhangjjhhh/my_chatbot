import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# ===================== 配置 =====================
API_KEY = "sk-wsthjcucuojcymksgvdovsrfdzxtvoybddrcjpcrkigltsqf"
BASE_URL = "https://api.siliconflow.cn/v1"
DOCS_FOLDER = "./docs"
PERSIST_DIR = "./chroma_db_multi"

# ===================== 1. 加载 & 初始化向量库 =====================
def init_vector_db():
    embedding = OpenAIEmbeddings(
        api_key=API_KEY,
        base_url=BASE_URL,
        model="BAAI/bge-m3"
    )

    # 如果库已存在，直接加载
    if os.path.exists(PERSIST_DIR):
        return Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embedding
        )

    # 不存在则创建
    docs = []
    for filename in os.listdir(DOCS_FOLDER):
        path = os.path.join(DOCS_FOLDER, filename)
        ext = os.path.splitext(filename)[1].lower()
        try:
            if ext in [".txt", ".md"]:
                loader = TextLoader(path, encoding="utf-8")
            elif ext == ".pdf":
                loader = PyPDFLoader(path)
            else:
                continue
            docs.extend(loader.load())
        except:
            continue

    splits = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=40
    ).split_documents(docs)

    db = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=PERSIST_DIR
    )

    return db

# ===================== 2. 全局初始化（只执行一次）=====================
db = init_vector_db()
llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model="deepseek-ai/DeepSeek-V3",
    temperature=0.1
)

# ===================== 3. RAG 查询函数（最简单、不报错）=====================
def rag_query(question, history_text):
    # 1. 检索（只传问题字符串，不会报错）
    retrieved = db.similarity_search(question, k=10)
    context = "\n".join([d.page_content for d in retrieved])

    # 2. 构建提示词
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是知识库助手，只根据资料回答，不编造。不知道就说无法回答。\n资料：{context}"),
        ("user", "历史对话：{history}\n问题：{question}")
    ])

    # 3. 调用模型
    chain = prompt | llm
    return chain.invoke({
        "context": context,
        "history": history_text,
        "question": question
    })

# ===================== 4. 主程序：对话 + 记忆 =====================
if __name__ == "__main__":
    print("✅ 带记忆的 RAG 机器人已启动（输入 exit 退出）\n")
    history = []

    while True:
        user_input = input("你：")
        if user_input.lower() == "exit":
            print("AI：再见！")
            break

        # 拼接历史
        history_text = "\n".join(history)

        # RAG 查询（绝对不报错）
        response = rag_query(user_input, history_text)
        ai_msg = response.content
        print("AI：", ai_msg)

        # 保存历史
        history.append(f"用户：{user_input}")
        history.append(f"AI：{ai_msg}")