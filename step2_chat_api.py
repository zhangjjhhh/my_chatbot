from fastapi import FastAPI
from pydantic import BaseModel
import requests
from typing import List, Optional

# ====================== 配置区 ======================
API_KEY = "sk-wsthjcucuojcymksgvdovsrfdzxtvoybddrcjpcrkigltsqf"
BASE_URL = "https://api.siliconflow.cn/v1/chat/completions"  # 正确地址
MODEL_NAME = "deepseek-ai/DeepSeek-V3"                      # 正确模型
# ====================================================

app = FastAPI(title="LLM 接口服务", version="1.0")

# 消息模型
class ChatMessage(BaseModel):
    role: str
    content: str

# 请求模型
class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

# 调用 LLM
def call_llm(messages, temperature=0.7):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"  # 必须带 Bearer，正确
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "temperature": temperature,
        "stream": False
    }
    try:
        response = requests.post(BASE_URL, headers=headers, json=payload)
        response.raise_for_status()  # 自动抛出 4xx/5xx 错误
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"LLM调用失败：{str(e)}"

# 接口
@app.post("/llm/chat")
def llm_chat(req: ChatRequest):
    reply = call_llm(req.messages, req.temperature)
    return {
        "code": 200,
        "result": reply
    }