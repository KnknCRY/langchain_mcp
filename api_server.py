from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from contextlib import asynccontextmanager
import json
import ast
import re
from decimal import Decimal
from typing import Optional, Dict, List, Any

# 全局變量存儲 LLM 和工具
llm_with_tools = None # 真正要用的 LLM物件
llm_chart_generator = None  # 專門用於生成圖表配置的 LLM 實例
client = None
tools_by_name = None  # 工具名稱到工具物件的map，用來找到對應的工具function


class ChatRequest(BaseModel):
    message: str
    system_prompt: str = """你是一個有用的助手，可以使用 PostgreSQL 數據庫工具來回答問題。

重要指示：
1. 當執行 SQL 查詢時，請直接調用 execute_sql 工具，不要添加額外的解釋性文字
2. SQL 查詢結果會自動進行視覺化處理，你只需要執行查詢即可
3. 如果用戶詢問數據查詢相關的問題，請直接使用 execute_sql 工具執行查詢
4. 對於查詢結果，簡潔地說明查詢內容即可，不需要重複顯示完整的數據表

這樣可以確保查詢結果能夠正確解析並生成圖表。""" # 系統提示(system prompt)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """初始化 LLM 和 MCP 客戶端"""
    global llm_with_tools, llm_chart_generator, client, tools_by_name

    # 初始化 Ollama 模型（用於對話和工具調用）
    llm = ChatOllama(
        model="qwen2.5:latest",
        base_url="http://localhost:11434"
    )

    # 初始化第二個 LLM 實例（專門用於生成圖表配置）
    llm_chart_generator = ChatOllama(
        model="qwen2.5:latest",
        base_url="http://localhost:11434"
    )
    print("✅ 圖表生成 LLM 實例已初始化")

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
    
    yield  # 應用運行期間
    
    # 清理資源
    print("正在清理資源...")

app = FastAPI(lifespan=lifespan)

# 添加 CORS 中間件，允許前端跨域訪問
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生產環境中應該設置具體的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def parse_sql_result(tool_result: Any) -> Optional[Dict[str, Any]]:
    """
    解析 SQL 查詢結果，提取結構化數據
    返回格式：
    {
        'columns': ['列名1', '列名2', ...],
        'rows': [['值1', '值2', ...], ...],
        'row_count': 行數
    }
    """
    if tool_result is None:
        return None
    
    # 優先檢查：如果已經是字典格式（包含 rows 和 columns）
    if isinstance(tool_result, dict):
        if 'rows' in tool_result and 'columns' in tool_result:
            return {
                'columns': tool_result['columns'],
                'rows': tool_result['rows'],
                'row_count': len(tool_result['rows'])
            }
        # 如果是單行結果（字典格式，鍵值對）
        elif all(isinstance(k, str) for k in tool_result.keys()):
            columns = list(tool_result.keys())
            rows = [[tool_result[col] for col in columns]]
            return {
                'columns': columns,
                'rows': rows,
                'row_count': 1
            }
    
    # 如果已經是列表格式（字典列表）
    if isinstance(tool_result, list) and len(tool_result) > 0:
        if isinstance(tool_result[0], dict):
            # 處理 Decimal 類型，轉換為 float
            def convert_value(val):
                if isinstance(val, Decimal):
                    return float(val)
                return val
            
            columns = list(tool_result[0].keys())
            rows = [[convert_value(row.get(col, '')) for col in columns] for row in tool_result]
            return {
                'columns': columns,
                'rows': rows,
                'row_count': len(rows)
            }
    
    result_str = str(tool_result)
    
    # 嘗試解析 Python 字典列表格式（如 "[{'id': 1, 'name': 'test'}]"）
    try:
        # 檢查是否是 Python 列表格式的字符串
        if isinstance(tool_result, str) and tool_result.strip().startswith('['):
            # 處理 Decimal('...') 格式，將其轉換為數字字符串
            import re
            def replace_decimal(match):
                decimal_str = match.group(1)
                try:
                    return decimal_str  # 直接返回引號內的數字字符串
                except:
                    return '0'
            
            # 將 Decimal('1000.000') 替換為 '1000.000' 或 1000.000
            cleaned_str = re.sub(r"Decimal\(['\"]([^'\"]+)['\"]\)", r'\1', tool_result)
            
            # 使用 ast.literal_eval 安全地解析 Python 字面量
            parsed = ast.literal_eval(cleaned_str)
            if isinstance(parsed, list) and len(parsed) > 0:
                if isinstance(parsed[0], dict):
                    # 轉換值為合適的類型
                    def convert_value(val):
                        if isinstance(val, str):
                            # 嘗試轉換為數字
                            try:
                                return float(val)
                            except ValueError:
                                return val
                        return val
                    
                    columns = list(parsed[0].keys())
                    rows = [[convert_value(row.get(col, '')) for col in columns] for row in parsed]
                    return {
                        'columns': columns,
                        'rows': rows,
                        'row_count': len(rows)
                    }
    except (ValueError, SyntaxError, TypeError) as e:
        pass
    
    # 嘗試解析 JSON 格式的結果
    try:
        # 如果是 JSON 字符串，嘗試解析
        if isinstance(tool_result, str) and (tool_result.strip().startswith('[') or tool_result.strip().startswith('{')):
            parsed = json.loads(tool_result)
            
            # 如果是列表，檢查是否為字典列表
            if isinstance(parsed, list) and len(parsed) > 0:
                if isinstance(parsed[0], dict):
                    columns = list(parsed[0].keys())
                    rows = [[row.get(col, '') for col in columns] for row in parsed]
                    return {
                        'columns': columns,
                        'rows': rows,
                        'row_count': len(rows)
                    }
                # 如果是列表的列表
                elif isinstance(parsed[0], list):
                    # 第一行可能是列名
                    if len(parsed) > 1:
                        columns = parsed[0] if all(isinstance(x, str) for x in parsed[0]) else [f'列{i+1}' for i in range(len(parsed[0]))]
                        rows = parsed[1:] if columns == parsed[0] else parsed
                        return {
                            'columns': columns,
                            'rows': rows,
                            'row_count': len(rows)
                        }
            # 如果是字典格式
            elif isinstance(parsed, dict):
                if 'rows' in parsed and 'columns' in parsed:
                    return {
                        'columns': parsed['columns'],
                        'rows': parsed['rows'],
                        'row_count': len(parsed['rows'])
                    }
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass
    
    # 嘗試解析表格格式的文本結果
    # 例如：| col1 | col2 |\n| val1 | val2 |
    lines = result_str.strip().split('\n')
    if len(lines) >= 2:
        # 尋找包含表格分隔符的行（跳過可能的文本前綴）
        table_start_idx = None
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if '|' in line_stripped:
                # 跳過分隔線（如 |----|----|）
                if not all(c in '-|: ' for c in line_stripped):
                    # 檢查是否包含多個 |（至少 2 個，表示是表格行）
                    pipe_count = line_stripped.count('|')
                    if pipe_count >= 2:
                        table_start_idx = i
                        break
        
        if table_start_idx is not None:
            # 解析列名（找到的第一個包含 | 的非分隔線行）
            header_line = lines[table_start_idx]
            columns = [col.strip() for col in header_line.split('|') if col.strip()]
            
            if columns:
                rows = []
                # 從下一行開始解析數據行
                for line in lines[table_start_idx + 1:]:
                    if '|' in line:
                        # 跳過分隔線
                        if not all(c in '-|: ' for c in line.strip()):
                            row = [cell.strip() for cell in line.split('|') if cell.strip()]
                            # 確保行數與列數匹配（可能需要調整）
                            if len(row) == len(columns):
                                rows.append(row)
                            elif len(row) > len(columns):
                                # 如果行數多於列數，截斷
                                rows.append(row[:len(columns)])
                
                if rows:
                    return {
                        'columns': columns,
                        'rows': rows,
                        'row_count': len(rows)
                    }
        
        # 嘗試解析 Tab 分隔的表格
        if '\t' in lines[0]:
            columns = [col.strip() for col in lines[0].split('\t') if col.strip()]
            if columns:
                rows = []
                for line in lines[1:]:
                    row = [cell.strip() for cell in line.split('\t') if cell.strip()]
                    if len(row) == len(columns):
                        rows.append(row)
                
                if rows:
                    return {
                        'columns': columns,
                        'rows': rows,
                        'row_count': len(rows)
                    }
    
    return None


def recommend_chart_type(parsed_data: Dict[str, Any]) -> str:
    """
    基於數據特徵推薦圖表類型
    返回推薦的圖表類型：'column', 'line', 'pie', 'bar', 'table'
    """
    if not parsed_data or 'columns' not in parsed_data:
        return 'table'
    
    columns = parsed_data['columns']
    rows = parsed_data.get('rows', [])
    row_count = parsed_data.get('row_count', 0)
    
    if row_count == 0:
        return 'table'
    
    # 如果只有一列，返回表格
    if len(columns) == 1:
        return 'table'
    
    # 檢查數據類型
    numeric_columns = []
    text_columns = []
    
    # 分析第一行數據來判斷類型
    if rows:
        first_row = rows[0]
        for i, col in enumerate(columns):
            if i < len(first_row):
                value = first_row[i]
                # 嘗試判斷是否為數值
                try:
                    float(str(value).replace(',', '').replace('$', '').replace('%', ''))
                    numeric_columns.append(col)
                except (ValueError, TypeError):
                    text_columns.append(col)
    
    # 如果有時間相關的列名，推薦折線圖
    time_keywords = ['date', 'time', '年月', '月份', '年', 'month', 'day', 'week']
    has_time_column = any(any(keyword in col.lower() for keyword in time_keywords) for col in columns)
    
    # 推薦邏輯
    if has_time_column and len(numeric_columns) > 0:
        return 'line'  # 時間序列數據用折線圖
    elif len(numeric_columns) == 1 and len(text_columns) >= 1:
        if row_count <= 10:
            return 'pie'  # 少量分類數據用餅圖
        else:
            return 'column'  # 多分類數據用柱狀圖
    elif len(numeric_columns) >= 2:
        return 'column'  # 多數值列用柱狀圖
    else:
        return 'table'  # 默認返回表格


def format_for_highcharts(parsed_data: Dict[str, Any], chart_type: str) -> Dict[str, Any]:
    """
    將解析後的數據格式化為 Highcharts 可用的格式
    """
    if not parsed_data:
        return {}
    
    columns = parsed_data['columns']
    rows = parsed_data.get('rows', [])
    
    if not rows:
        return {}
    
    # 識別分類軸和數值軸
    numeric_indices = []
    category_index = 0
    
    # 嘗試找出數值列
    for i, col in enumerate(columns):
        try:
            # 檢查第一行數據
            if rows and i < len(rows[0]):
                float(str(rows[0][i]).replace(',', '').replace('$', '').replace('%', ''))
                numeric_indices.append(i)
        except (ValueError, TypeError):
            if i == 0:
                category_index = i
    
    # 如果沒有找到數值列，使用第二列開始的所有列
    if not numeric_indices and len(columns) > 1:
        numeric_indices = list(range(1, len(columns)))
    
    if not numeric_indices:
        return {
            'categories': [],
            'series': []
        }
    
    # 構建分類軸（通常是第一列）
    categories = []
    if category_index < len(columns):
        categories = [str(row[category_index]) if category_index < len(row) else '' for row in rows]
    
    # 構建數據系列
    series = []
    for idx in numeric_indices:
        if idx < len(columns):
            series_name = columns[idx]
            series_data = []
            for row in rows:
                if idx < len(row):
                    value = row[idx]
                    try:
                        # 嘗試轉換為數值
                        numeric_value = float(str(value).replace(',', '').replace('$', '').replace('%', ''))
                        series_data.append(numeric_value)
                    except (ValueError, TypeError):
                        series_data.append(0)
            
            series.append({
                'name': series_name,
                'data': series_data
            })
    
    result = {
        'categories': categories if categories else [f'項目{i+1}' for i in range(len(rows))],
        'series': series
    }
    
    # 根據圖表類型調整格式
    if chart_type == 'pie':
        # 餅圖需要特殊格式
        if series:
            pie_data = []
            for i, category in enumerate(result['categories']):
                if i < len(series[0]['data']):
                    pie_data.append({
                        'name': category,
                        'y': series[0]['data'][i]
                    })
            result = {
                'series': [{
                    'name': series[0]['name'] if series else '數據',
                    'data': pie_data
                }]
            }
    
    return result


def extract_json_from_llm_response(response_text: str) -> Optional[Dict[str, Any]]:
    """
    從 LLM 回應中提取 JSON，處理可能包含 markdown 代碼塊的情況
    """
    if not response_text:
        return None
    
    # 嘗試直接解析 JSON
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        pass
    
    # 嘗試提取 markdown 代碼塊中的 JSON
    # 匹配 ```json ... ``` 或 ``` ... ```
    json_patterns = [
        r'```json\s*\n(.*?)\n```',  # ```json ... ```
        r'```\s*\n(.*?)\n```',      # ``` ... ```
        r'```json\s*(.*?)```',      # ```json ... ``` (單行)
        r'```\s*(.*?)```',          # ``` ... ``` (單行)
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, response_text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue
    
    # 嘗試找到第一個 { 到最後一個 } 之間的內容
    first_brace = response_text.find('{')
    last_brace = response_text.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_str = response_text[first_brace:last_brace + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    return None


async def generate_chart_config_with_llm(tool_result: Any) -> Optional[Dict[str, Any]]:
    """
    使用 LLM 分析 SQL 查詢結果並生成 Highcharts 配置
    
    返回格式：
    {
        'chart_data': {
            'categories': [...],
            'series': [...]
        },
        'recommended_chart_type': 'column' | 'line' | 'pie' | 'bar' | 'table',
        'columns': [...],
        'rows': [...],
        'row_count': int
    }
    """
    global llm_chart_generator
    
    if llm_chart_generator is None:
        print("❌ 圖表生成 LLM 實例未初始化")
        return None
    
    # 將查詢結果轉換為字符串格式供 LLM 分析
    # 如果是複雜對象，先嘗試轉換為 JSON
    try:
        if isinstance(tool_result, (dict, list)):
            data_str = json.dumps(tool_result, ensure_ascii=False, default=str)
        else:
            data_str = str(tool_result)
    except Exception as e:
        print(f"❌ 無法序列化查詢結果: {e}")
        data_str = str(tool_result)
    
    # 設計詳細的 system prompt
    system_prompt = """你是一個數據視覺化專家，專門分析 SQL 查詢結果並生成 Highcharts 圖表配置。

你的任務：
1. 分析提供的 SQL 查詢結果數據
2. 識別數據結構（列名、數據類型、行數等）
3. 選擇最合適的圖表類型（column, line, pie, bar, table）
4. 生成符合 Highcharts 格式的 JSON 配置

圖表類型選擇規則：
- column（柱狀圖）：用於比較不同類別的數值，特別是時間序列或多個數值列
- line（折線圖）：用於顯示趨勢，特別是包含時間/日期列的數據
- pie（餅圖）：用於顯示部分與整體的關係，適合少量分類（≤10個）且只有一個數值列
- bar（條形圖）：與柱狀圖類似，但橫向顯示
- table（表格）：當數據不適合圖表化，或只有一列時使用

輸出格式要求：
你必須返回一個有效的 JSON 對象，格式如下：
{
    "chart_data": {
        "categories": ["類別1", "類別2", ...],  // 用於 x 軸的分類（可選，pie 圖不需要）
        "series": [
            {
                "name": "系列名稱",
                "data": [數值1, 數值2, ...]  // 對於 pie 圖，data 格式為 [{"name": "類別", "y": 數值}, ...]
            }
        ]
    },
    "recommended_chart_type": "column" | "line" | "pie" | "bar" | "table",
    "columns": ["列名1", "列名2", ...],
    "rows": [["值1", "值2", ...], ...],
    "row_count": 行數
}

重要提示：
- 必須返回純 JSON，不要包含 markdown 代碼塊標記
- 確保所有數值都是數字類型，不是字符串
- 對於 pie 圖，series[0].data 應該是對象數組，每個對象包含 "name" 和 "y" 字段
- 如果數據不適合視覺化，recommended_chart_type 設為 "table"
- 確保 JSON 格式正確且完整"""

    user_prompt = f"""請分析以下 SQL 查詢結果並生成 Highcharts 配置：

查詢結果數據：
{data_str}

請返回符合上述格式的 JSON 配置。"""

    try:
        # 調用 LLM 生成配置
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = await llm_chart_generator.ainvoke(messages)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        print(f"📝 LLM 原始回應長度: {len(response_text)} 字符")
        
        # 提取 JSON
        chart_config = extract_json_from_llm_response(response_text)
        
        if chart_config:
            # 如果 LLM 返回的配置缺少 columns 或 rows，嘗試從原始數據中提取
            if 'columns' not in chart_config or 'rows' not in chart_config:
                parsed_data = parse_sql_result(tool_result)
                if parsed_data:
                    if 'columns' not in chart_config:
                        chart_config['columns'] = parsed_data.get('columns', [])
                    if 'rows' not in chart_config:
                        chart_config['rows'] = parsed_data.get('rows', [])
                    if 'row_count' not in chart_config:
                        chart_config['row_count'] = parsed_data.get('row_count', 0)
            
            # 確保有 row_count
            if 'row_count' not in chart_config:
                if 'rows' in chart_config:
                    chart_config['row_count'] = len(chart_config['rows'])
                else:
                    chart_config['row_count'] = 0
            
            print("✅ LLM 成功生成圖表配置")
            return chart_config
        else:
            print(f"❌ 無法從 LLM 回應中提取 JSON，回應前500字符: {response_text[:500]}")
            return None
            
    except Exception as e:
        print(f"❌ LLM 生成圖表配置時發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return None


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
            # chunck物件會長這樣
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

        # 執行工具調用，這邊可以ctrl+F尋找”chunck物件會長這樣“，可以看到tool_call具體物件內容長什麼樣子
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

            # 如果是 SQL 查詢工具，解析結果並發送結構化數據
            if tool_name == 'execute_sql' or 'sql' in tool_name.lower():
                # 檢查是否為錯誤訊息，如果是則跳過 LLM 生成
                tool_result_str = str(tool_result).strip()
                is_error = (
                    tool_result_str.startswith('Error:') or 
                    tool_result_str.startswith('錯誤:') or
                    'error' in tool_result_str.lower()[:50] or
                    'does not exist' in tool_result_str.lower() or
                    '不存在' in tool_result_str
                )
                
                if is_error:
                    # 錯誤訊息，直接使用後備解析邏輯
                    print("⚠️ 檢測到 SQL 錯誤，跳過 LLM 生成，使用後備解析邏輯")
                    parsed_data = parse_sql_result(tool_result)
                    if parsed_data:
                        print(f"✅ 成功解析 SQL 結果: {len(parsed_data.get('columns', []))} 列, {parsed_data.get('row_count', 0)} 行")
                        recommended_type = recommend_chart_type(parsed_data)
                        print(f"📊 推薦圖表類型: {recommended_type}")
                        chart_data = format_for_highcharts(parsed_data, recommended_type)
                        print(f"📈 圖表數據準備完成: {len(chart_data.get('series', []))} 個系列")
                        
                        chart_event = {
                            'type': 'chart_data',
                            'tool_name': tool_name,
                            'columns': parsed_data['columns'],
                            'rows': parsed_data['rows'],
                            'row_count': parsed_data['row_count'],
                            'chart_data': chart_data,
                            'recommended_chart_type': recommended_type
                        }
                        yield f"data: {json.dumps(chart_event, ensure_ascii=False)}\n\n"
                        print("✅ 圖表數據已發送（後備解析）")
                    else:
                        print(f"❌ 無法解析 SQL 結果，結果類型: {type(tool_result)}, 前200字符: {str(tool_result)[:200]}")
                else:
                    # 正常結果，優先使用 LLM 生成圖表配置
                    llm_chart_config = await generate_chart_config_with_llm(tool_result)
                    
                    # 驗證 LLM 返回的配置是否有效（檢查是否有實際數據）
                    if llm_chart_config and 'chart_data' in llm_chart_config:
                        # 檢查是否為虛假的示例數據
                        chart_data = llm_chart_config.get('chart_data', {})
                        columns = llm_chart_config.get('columns', [])
                        rows = llm_chart_config.get('rows', [])
                        
                        # 如果 columns 或 rows 看起來像示例數據（包含 "Column", "Value", "Category" 等），則視為無效
                        is_example_data = (
                            any('Column' in str(col) or 'Category' in str(col) for col in columns) or
                            any('Value' in str(row) for row in rows[:3] if isinstance(row, (list, str)))
                        )
                        
                        if is_example_data and not isinstance(tool_result, (dict, list)):
                            # LLM 生成了示例數據但原始數據不是結構化的，視為失敗
                            print("⚠️ LLM 返回了示例數據，視為生成失敗，使用後備解析邏輯")
                            llm_chart_config = None
                    
                    if llm_chart_config and 'chart_data' in llm_chart_config:
                        # LLM 成功生成配置
                        print(f"✅ LLM 成功生成圖表配置: {llm_chart_config.get('recommended_chart_type', 'unknown')}")
                        
                        chart_event = {
                            'type': 'chart_data',
                            'tool_name': tool_name,
                            'columns': llm_chart_config.get('columns', []),
                            'rows': llm_chart_config.get('rows', []),
                            'row_count': llm_chart_config.get('row_count', 0),
                            'chart_data': llm_chart_config.get('chart_data', {}),
                            'recommended_chart_type': llm_chart_config.get('recommended_chart_type', 'table')
                        }
                        yield f"data: {json.dumps(chart_event, ensure_ascii=False)}\n\n"
                        print("✅ 圖表數據已發送（LLM 生成）")
                    else:
                        # LLM 生成失敗，回退到原有的解析邏輯
                        print("⚠️ LLM 生成失敗，使用原有解析邏輯作為後備")
                        parsed_data = parse_sql_result(tool_result)
                        if parsed_data:
                            print(f"✅ 成功解析 SQL 結果: {len(parsed_data.get('columns', []))} 列, {parsed_data.get('row_count', 0)} 行")
                            recommended_type = recommend_chart_type(parsed_data)
                            print(f"📊 推薦圖表類型: {recommended_type}")
                            chart_data = format_for_highcharts(parsed_data, recommended_type)
                            print(f"📈 圖表數據準備完成: {len(chart_data.get('series', []))} 個系列")
                            
                            chart_event = {
                                'type': 'chart_data',
                                'tool_name': tool_name,
                                'columns': parsed_data['columns'],
                                'rows': parsed_data['rows'],
                                'row_count': parsed_data['row_count'],
                                'chart_data': chart_data,
                                'recommended_chart_type': recommended_type
                            }
                            yield f"data: {json.dumps(chart_event, ensure_ascii=False)}\n\n"
                            print("✅ 圖表數據已發送（後備解析）")
                        else:
                            print(f"❌ 無法解析 SQL 結果，結果類型: {type(tool_result)}, 前200字符: {str(tool_result)[:200]}")

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
    return {
        "status": "ok",
        "llm_ready": llm_with_tools is not None,
        "chart_llm_ready": llm_chart_generator is not None
    }


@app.get("/")
async def serve_index():
    """提供前端頁面"""
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "前端頁面未找到，請確保 index.html 文件存在"}

# 執行方式
# .venv/bin/python api_server.py
# curl -X POST http://localhost:9999/chat/stream \
#     -H "Content-Type: application/json" \
#     -d '{"message": "撈出"1_purchase_order_detail"資料表所有資料"}' \
#     --no-buffer
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9999)
