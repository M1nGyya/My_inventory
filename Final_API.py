import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
# 从FInal_RAG模块导入查询函数
from Final_RAG import query_rag_system

# 初始化 FastAPI 应用
app = FastAPI(
    title="智能法律问答 RAG 系统 API",
    description="一个基于 FastAPI 和 RAG 架构的后端服务",
    version="1.0.0"
)

# 配置 CORS 中间件，允许来自任何源的请求
# 用于连接 Gradio 前端
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 请求体模型，API输入
class QueryRequest(BaseModel):
    query: str

# 响应体模型
class QueryResponse(BaseModel):
    answer: str

@app.post("/api/chat", response_model=QueryResponse, summary="接收问题并返回答案")
def chat_endpoint(request: QueryRequest):
    # 调用RAG处理查询
    response_text = query_rag_system(request.query)
    # 返回标准化的JSON响应
    return {"answer": response_text}


# 使得该文件可以直接通过 'python api.py' 运行 (主要用于开发)
if __name__ == "__main__":
    print("启动 FastAPI 服务器...")
    uvicorn.run(app, host="0.0.0.0", port=8000)