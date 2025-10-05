from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
import json

app = FastAPI()

# 全局變量存儲 LLM 和工具
llm_with_tools = None # 真正要用的 LLM物件
client = None


class ChatRequest(BaseModel):
    message: str
    system_prompt: str = "你是一個有用的助手，可以使用 PostgreSQL 數據庫工具來回答問題。"


@app.on_event("startup")
async def startup_event():
    """初始化 LLM 和 MCP 客戶端"""
    global llm_with_tools, client

    # 初始化 Ollama 模型
    llm = ChatOllama(
        model="qwen2.5:1.5b",
        base_url="http://localhost:11434"
    )

    # 連接到 MCP 服務
    print("正在連接到 MCP 服務...")
    client = MultiServerMCPClient(
        connections={
            "postgres": {
                "transport": "sse",
                "url": "http://localhost:8000/sse",
                "timeout": 10.0,
                "sse_read_timeout": 300.0
            }
        }
    )

    # 載入可用的工具
    print("正在載入 MCP 工具...")
    tools = await client.get_tools()
    print(f"已載入 {len(tools)} 個 MCP 工具")

    # 將 MCP 工具綁定到 LLM
    llm_with_tools = llm.bind_tools(tools)


async def generate_stream(message: str, system_prompt: str):
    """生成串流回應"""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=message)
    ]

    async for chunk in llm_with_tools.astream(messages):
        # 構建回應數據
        response_data = {
            "content": chunk.content if chunk.content else "",
            "tool_calls": []
        }

        # 檢查是否有工具調用
        if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
            response_data["tool_calls"] = [
                {
                    "name": tc.get("name"),
                    "args": tc.get("args"),
                    "id": tc.get("id")
                }
                for tc in chunk.tool_calls
            ]

        # 使用 SSE 格式發送數據
        yield f"data: {json.dumps(response_data, ensure_ascii=False)}\n\n"

    # 發送結束標記
    yield "data: [DONE]\n\n"


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """聊天 API - 串流回應"""
    return StreamingResponse(
        generate_stream(request.message, request.system_prompt),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/health")
async def health():
    """健康檢查"""
    return {"status": "ok", "llm_ready": llm_with_tools is not None}

# .venv/bin/python api_server.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
