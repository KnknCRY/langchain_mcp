import asyncio
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient


async def main():
    # 初始化 Ollama 模型
    llm = ChatOllama(
        model="qwen2.5:latest",
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
    print(f"\n可用的 MCP 工具: {len(tools)} 個")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")

    # 將 MCP 工具綁定到 LLM
    llm_with_tools = llm.bind_tools(tools)

    # 進行簡單對話
    print("\n開始對話...")
    messages = [
        SystemMessage(content="你是一個有用的助手，可以使用 PostgreSQL 數據庫工具來回答問題。"),
        HumanMessage(content="撈出sales.customers資料表所有資料")
    ]

    # 建立工具名稱到工具物件的映射
    tools_by_name = {tool.name: tool for tool in tools}

    # 工具調用循環，最多執行 10 次以防無限循環
    for iteration in range(10):
        print(f"\n=== 迭代 {iteration + 1} ===")

        # 調用 LLM
        response = await llm_with_tools.ainvoke(messages)
        messages.append(response)

        # 顯示 LLM 回應
        if response.content:
            print(f"\n助手回應: {response.content}")

        # 檢查是否有工具調用
        if not response.tool_calls:
            print("\n對話完成（無需調用工具）")
            break

        # 執行工具調用
        print(f"\n檢測到 {len(response.tool_calls)} 個工具調用")
        for tool_call in response.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']

            print(f"\n調用工具: {tool_name}")
            print(f"參數: {tool_args}")

            # 找到對應的工具並執行
            if tool_name in tools_by_name:
                tool = tools_by_name[tool_name]
                tool_result = await tool.ainvoke(tool_args)
            else:
                tool_result = f"錯誤: 找不到工具 '{tool_name}'"

            print(f"工具結果: {tool_result[:200]}..." if len(str(tool_result)) > 200 else f"工具結果: {tool_result}")

            # 將工具結果加入訊息歷史
            from langchain_core.messages import ToolMessage
            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call['id']
                )
            )
    else:
        print("\n達到最大迭代次數")

    print()  # 換行


# 執行方式：.venv/bin/python chat_with_ollama.py
if __name__ == "__main__":
    asyncio.run(main())
