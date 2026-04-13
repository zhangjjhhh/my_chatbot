from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

# 配置硅基流动大模型
llm = ChatOpenAI(
    api_key="sk-wsthjcucuojcymksgvdovsrfdzxtvoybddrcjpcrkigltsqf",  # 替换成你的硅基API Key
    base_url="https://api.siliconflow.cn/v1",
    model="deepseek-ai/DeepSeek-V3",
    temperature=0.1
)

# ===================== 2. 带记忆的提示词模板 =====================
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的AI助手，记得所有对话历史。"),
    # 👇 这里就是【记忆存储区】，自动保存历史对话
    MessagesPlaceholder(variable_name="history"),
    ("user", "{input}")
])

# 链式调用
chain = prompt | llm

# ===================== 4. 开启记忆功能 =====================
# 内存记忆（关掉程序消失，适合测试）
store = {}
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 给链加上记忆
chat_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)
# ===================== 5. 开始多轮对话 =====================
print("=== 带记忆的聊天机器人（输入 exit 退出）===")
while True:
    user_input = input("你：")
    if user_input.lower() == "exit":
        print("AI：再见！")
        break

    # 发送消息（session_id 用来区分不同用户）
    ai_response = chat_with_memory.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": "zhangjiahao_123"}}
    )

    print("AI：", ai_response.content)