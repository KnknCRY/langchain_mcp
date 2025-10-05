import asyncio
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient


async def main():
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
    print(f"\n可用的 MCP 工具: {len(tools)} 個")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")

    # 將 MCP 工具綁定到 LLM
    llm_with_tools = llm.bind_tools(tools)

    # 進行簡單對話
    print("\n開始對話...")
    messages = [
        SystemMessage(content="你是一個有用的助手，可以使用 PostgreSQL 數據庫工具來回答問題。"),
        HumanMessage(content="你好！請介紹一下你自己，並告訴我你可以使用哪些工具。")
    ]

    response = await llm_with_tools.ainvoke(messages)
    print(f"\n助手回應:\n{response.content}")

    # 如果有工具調用
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"\n工具調用: {response.tool_calls}")


# 執行方式：.venv/bin/python chat_with_ollama.py
if __name__ == "__main__":
    asyncio.run(main())
