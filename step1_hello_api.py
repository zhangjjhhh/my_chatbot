from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="My Frist API", version="1.0")

class CalculatopRequest(BaseModel):
    expression: str

@app.get("/",summary="根路径：Hello World")
def read_root():
    return {"message": "Hello World","status":"success"}

@app.get("/hello/{name}",summary="打招呼接口")
def hello_name(name: str):
    return {"message":f"Hello {name}","status":"success"}

@app.post("/calculator",summary="简单计算器接口")
def calculator(req: CalculatopRequest):
    try:
        allowed_chars = "0123456789+-*/(). "
        if not all(c in allowed_chars for c in req.expression):
            return {"code":400,"message":"仅支持四则运算","result":None}

        result = eval(req.expression, {"__builtins__":None},{})
        return {"code":200,"message":"计算成功","result":result}
    except Exception as e:
        return {"code":500,"message":f"计算失败:{str(e)}","result":None}
