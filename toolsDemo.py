import json
import requests
from datetime import datetime

# ====================
# 硅基 API KEY
# ====================
API_KEY = "sk-wsthjcucuojcymksgvdovsrfdzxtvoybddrcjpcrkigltsqf"
BASE_URL = "https://api.siliconflow.cn/v1/chat/completions"
MODEL_NAME = "deepseek-ai/DeepSeek-V3"

# ====================
# 工具定义
# ====================
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "获取当前系统日期和时间",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名，例如：北京"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "两个数字的加减乘除计算",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "数字1"},
                    "b": {"type": "number", "description": "数字2"},
                    "op": {"type": "string", "enum": ["+", "-", "*", "/"]}
                },
                "required": ["a", "b", "op"]
            }
        }
    }
]

# ====================
# 工具实现
# ====================
def get_time():
    return datetime.now().strftime("当前时间：%Y-%m-%d %H:%M:%S")

def get_weather(city):
    return f"{city}：晴，22℃，微风"

def calculate(a, b, op):
    try:
        if op == "+":
            return a + b
        elif op == "-":
            return a - b
        elif op == "*":
            return a * b
        elif op == "/":
            return a / b if b != 0 else "除数不能为0"
        else:
            return "不支持的运算"
    except:
        return "计算错误"

# ====================
# 统一执行工具
# ====================
def execute_tool(func_name, args):
    try:
        if func_name == "get_time":
            return get_time()
        elif func_name == "get_weather":
            return get_weather(args.get("city"))
        elif func_name == "calculate":
            return calculate(args.get("a"), args.get("b"), args.get("op"))
        else:
            return "未知工具"
    except Exception as e:
        return f"工具执行失败：{str(e)}"

# ====================
# 核心：调用大模型（带错误处理）
# ====================
def call_model(messages, tools=None):
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.1
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"  # 硅基必须加这个

    try:
        resp = requests.post(
            url=BASE_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json=payload
        )
        return resp.json()
    except Exception as e:
        print(f"请求失败：{e}")
        return None

# ====================
# 完整聊天机器人
# ====================
def chat_bot(user_input):
    messages = [{"role": "user", "content": user_input}]

    # 第一次请求：判断工具
    resp = call_model(messages, tools)

    if not resp:
        return "模型请求失败"

    # 打印返回结果（调试用）
    # print("模型返回：", json.dumps(resp, indent=2, ensure_ascii=False))

    # 处理错误
    if "error" in resp:
        return f"API错误：{resp['error']['message']}"

    # 解析消息
    msg = resp["choices"][0]["message"]
    messages.append(msg)

    # 如果需要调用工具
    if "tool_calls" in msg:
        tool = msg["tool_calls"][0]["function"]
        func_name = tool["name"]
        args = json.loads(tool["arguments"])

        # 执行工具
        tool_result = execute_tool(func_name, args)

        # 把结果返回给模型
        messages.append({
            "role": "tool",
            "content": str(tool_result)
        })

        # 第二次请求：生成回答
        final_resp = call_model(messages)
        if "choices" in final_resp:
            return final_resp["choices"][0]["message"]["content"]
        else:
            return str(tool_result)
    else:
        return msg["content"]

# ====================
# 测试区
# ====================
if __name__ == "__main__":
    #print(chat_bot("现在几点了？"))
    #print(chat_bot("北京天气怎么样？"))
    print(chat_bot("33 + 66 等于多少？"))