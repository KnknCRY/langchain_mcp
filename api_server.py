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
tools_by_name = None  # 工具名稱到工具物件的map，用來找到對應的工具function


class ChatRequest(BaseModel):
    message: str
    system_prompt: str = "你是一個有用的助手，可以使用 PostgreSQL 數據庫工具來回答問題。" # 系統提示(system prompt)


@app.on_event("startup")
async def startup_event():
    """初始化 LLM 和 MCP 客戶端"""
    global llm_with_tools, client, tools_by_name

    # 初始化 Ollama 模型
    llm = ChatOllama(
        model="qwen2.5:latest",
        base_url="http://localhost:11434"
    )

    # 連接到 MCP 服務
    # 用下面這句docker跑mcp server，記得要改DATABASE_URI
    # docker run -p 8000:8000 -e DATABASE_URI=postgresql://postgres:50984878@localhost:5432/sales_db crystaldba/postgres-mcp --access-mode=unrestricted --transport=sse
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

    # 建立工具名稱到工具物件的map，用來找到對應的工具function
    tools_by_name = {tool.name: tool for tool in tools}

    # 將 MCP 工具綁定到 LLM
    llm_with_tools = llm.bind_tools(tools)


async def generate_stream(message: str, system_prompt: str):
    """生成串流回應"""
    from langchain_core.messages import ToolMessage

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=message)
    ]

    # 工具調用循環，最多執行 10 次以防無限循環
    # 每一次LLM回應都會判斷要不要調用工具，防止無限掉用工具的循環
    for iteration in range(10):
        # 發送迭代信息(這個不是很重要，只是會用chunk回傳目前迭代的次數而已)
        yield f"data: {json.dumps({'type': 'iteration', 'iteration': iteration + 1}, ensure_ascii=False)}\n\n"

        # 調用 LLM (使用串流)
        response_content = ""
        response_tool_calls = []

        async for chunk in llm_with_tools.astream(messages):
            # print("串流內容:", chunk, flush=True)
            # 串流內容會長這樣
            """
                串流內容: content='这个' additional_kwargs={} response_metadata={} id='run--2b83d8e2-a1ef-4d63-9198-6acb34323e8a'
                串流內容: content='查询' additional_kwargs={} response_metadata={} id='run--2b83d8e2-a1ef-4d63-9198-6acb34323e8a'
                串流內容: content='。\n' additional_kwargs={} response_metadata={} id='run--2b83d8e2-a1ef-4d63-9198-6acb34323e8a'
                串流內容: content='' additional_kwargs={} response_metadata={} id='run--2b83d8e2-a1ef-4d63-9198-6acb34323e8a' tool_calls=[{'name': 'execute_sql', 'args': {'sql': 'SELECT * FROM sales.customers'}, 'id': '7b1e1fad-1e85-4e74-afe3-269c1be2fa20', 'type': 'tool_call'}] tool_call_chunks=[{'name': 'execute_sql', 'args': '{"sql": "SELECT * FROM sales.customers"}', 'id': '7b1e1fad-1e85-4e74-afe3-269c1be2fa20', 'index': None, 'type': 'tool_call_chunk'}]
                串流內容: content='' additional_kwargs={} response_metadata={'model': 'qwen2.5:latest', 'created_at': '2025-10-07T03:18:16.47494Z', 'done': True, 'done_reason': 'stop', 'total_duration': 4421769125, 'load_duration': 55672125, 'prompt_eval_count': 1177, 'prompt_eval_duration': 1559918875, 'eval_count': 55, 'eval_duration': 2802383875, 'model_name': 'qwen2.5:latest'} id='run--2b83d8e2-a1ef-4d63-9198-6acb34323e8a' usage_metadata={'input_tokens': 1177, 'output_tokens': 55, 'total_tokens': 1232}
            """
            if chunk.content:
                response_content += chunk.content
                yield f"data: {json.dumps({'type': 'content', 'content': chunk.content}, ensure_ascii=False)}\n\n"

            # 如果LLM回應有工具調用，不會有content，而是要找有沒有tool_calls這個屬性，若有則把該工具調用的function存到陣列中
            if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                response_tool_calls.extend(chunk.tool_calls)

        # 構建完整的 response 物件加入訊息歷史，每次迭代都會把這次的回應內容和工具調用加入訊息歷史，供下一次迭代讓LLM參考
        from langchain_core.messages import AIMessage
        response = AIMessage(content=response_content, tool_calls=response_tool_calls)
        messages.append(response)

        # 檢查是否有工具調用，當沒有工具要調用代表對話完成，直接break跳出迴圈，代表此次對話結束
        if not response_tool_calls:
            yield f"data: {json.dumps({'type': 'done', 'message': '對話完成'}, ensure_ascii=False)}\n\n"
            break

        # 發送工具調用信息
        yield f"data: {json.dumps({'type': 'tool_calls_detected', 'count': len(response_tool_calls)}, ensure_ascii=False)}\n\n"

        # 執行工具調用，這邊可以ctrl+F尋找”串流內容會長這樣“，可以看到tool_call具體物件內容長什麼樣子
        for tool_call in response_tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']

            # 發送工具調用開始
            yield f"data: {json.dumps({'type': 'tool_call_start', 'name': tool_name, 'args': tool_args}, ensure_ascii=False)}\n\n"

            # 找到對應的工具並執行
            if tool_name in tools_by_name:
                tool = tools_by_name[tool_name]
                tool_result = await tool.ainvoke(tool_args)
            else:
                tool_result = f"錯誤: 找不到工具 '{tool_name}'"

            # 發送工具執行結果
            result_preview = str(tool_result)[:200] + ("..." if len(str(tool_result)) > 200 else "")
            yield f"data: {json.dumps({'type': 'tool_result', 'name': tool_name, 'result': result_preview}, ensure_ascii=False)}\n\n"

            # 將工具結果加入訊息歷史
            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call['id']
                )
            )
    else:
        yield f"data: {json.dumps({'type': 'max_iterations', 'message': '達到最大迭代次數'}, ensure_ascii=False)}\n\n"

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

# 執行方式
# .venv/bin/python api_server.py
# curl -X POST http://localhost:8080/chat/stream \
#     -H "Content-Type: application/json" \
#     -d '{"message": "撈出sales.customers資料表所有資料"}' \
#     --no-buffer
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
