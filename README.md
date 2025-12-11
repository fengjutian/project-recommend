把你的文档放到 ./docs/（支持 .txt, .md, .pdf）。

安装依赖：

pip install -r requirements.txt


在项目根创建 .env 并填入 OPENAI_API_KEY。

第一次运行会把 docs 全量索引并在 ./chroma_db 生成向量库：

python app.py
# 或者：
uvicorn app:app --host 0.0.0.0 --port 8000 --reload


测试调用（curl）：

curl -X POST "http://127.0.0.1:8000/query" -H "Content-Type: application/json" -d '{"session_id":"s1","question":"产品 X 的目标用户是谁？"}'

返回示例（JSON）包含 answer, sources（每个 chunk 的来源路径 & chunk idx）和 rerank_scores（LLM 给每个候选的评分）。

多轮对话：保持同一个 session_id（例如 "s1"），服务会在 prompt 里加入近期对话记忆（最多 MEMORY_MAX_TURNS 轮），从而实现上下文连续性。若想清会话可以 POST /reset_session。

