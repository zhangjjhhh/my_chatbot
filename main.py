from openai import OpenAI
from dotenv import load_dotenv
import os
import json

# 加载配置
load_dotenv()
client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL")
)

prompt = """
你是一个严格的信息提取助手。
规则：
1. 只返回标准 JSON，不输出任何多余文字
2. 不要加解释、不要前言后语、不要```json代码块
3. 严格按照下面字段输出：name, age, city, job

用户输入：李四，25岁，来自成都，是一名程序员。
"""
# 初始化对话历史（system角色设定AI身份）
messages = [
    {"role":"system","content":"你是一名信息提取助手"},
    {"role": "user","content": prompt}
]

def chat():
    print("✨ AI 聊天机器人已启动（输入 quit 退出）")
    while True:
        # 1. 获取用户输入
        user_input = input("你想问AI什么：").strip()
        # 在用户输入判断里加一行
        if user_input.lower() == "clear":
            messages.clear()
            messages.append({
                "role": "system",
                "content": "你是专业信息处理的助手，只输出JSON"
            })
            print("已清空记忆\n")
            continue

        # 2. 退出机制
        if user_input == "quit": break

        # 3. 把用户输入加入对话历史
        #messages.append({"role": "user", "content": user_input})

        # 4. 发送完整对话历史给大模型
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3",
            messages=messages,
            stream=False    # 新手先用False
        )

        # 4. 获取返回结果
        result_text = response.choices[0].message.content.strip()

        # 5. 解析 JSON（Agent 开发核心步骤）
        data = json.loads(result_text)

        # 6. 打印结果
        print("解析后的JSON数据：")
        print(data)
        print("-" * 40)
        print(f"姓名：{data['name']}")
        print(f"年龄：{data['age']}")
        print(f"城市：{data['city']}")
        print(f"职业：{data['job']}")
# 启动聊天
if __name__ == "__main__":
    chat()