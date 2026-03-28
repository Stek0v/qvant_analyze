"""Async adapter для Levara vector search (REST API). Замена Cognee.

Levara: 2.6ms p50 latency, HYBRID search, HNSW vectors + BM25.
Endpoint: POST /api/v1/search (requires vector), or MCP tools via Claude.
Исполь��уем MCP-compatible REST: /api/v1/search с pre-embedded query.
"""

from __future__ import annotations

import logging

import httpx

from router_api.models import RetrievalResult

logger = logging.getLogger(__name__)


class LevaraAdapter:
    def __init__(self, base_url: str = "http://10.23.0.53:8080",
                 collection: str = "qvant", timeout: float = 10.0):
        self._base_url = base_url
        self._collection = collection
        self._timeout = timeout

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5) as c:
                r = await c.get(f"{self._base_url}/health")
                return r.status_code == 200
        except Exception:
            return False

    async def search(
        self,
        query: str,
        mode: str = "shallow",
        search_type: str = "HYBRID",
        collection: str | None = None,
    ) -> list[RetrievalResult]:
        """Поиск через Levara REST API.

        Levara требует вектор для /api/v1/search, поэтому используем
        /api/v1/search/text endpoint (есл�� доступен) или MCP search tool.
        Fallback: embed через llama-server embeddings → search by vector.
        """
        top_k = 5 if mode == "shallow" else 20
        col = collection or self._collection

        try:
            # Попробуем text search endpoint
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                r = await client.post(
                    f"{self._base_url}/api/v1/search/text",
                    json={
                        "query": query,
                        "collection": col,
                        "search_type": search_type,
                        "top_k": top_k,
                    },
                )
                if r.status_code == 200:
                    return self._parse_results(r.json())

                # Fallback: попробуем MCP-style endpoint
                r = await client.post(
                    f"{self._base_url}/mcp",
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/call",
                        "params": {
                            "name": "search",
                            "arguments": {
                                "search_query": query,
                                "collection": col,
                                "search_type": search_type,
                                "top_k": top_k,
                            },
                        },
                    },
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json, text/event-stream",
                    },
                )
                if r.status_code == 200:
                    return self._parse_mcp_response(r.text)

        except Exception as e:
            logger.warning(f"Levara search failed: {e}")

        return []

    @staticmethod
    def _parse_results(data: dict) -> list[RetrievalResult]:
        results = []
        for item in data.get("results", []):
            content = item.get("content", item.get("text", item.get("value", "")))
            score = float(item.get("score", item.get("similarity", 0.5)))
            source = item.get("collection", item.get("source", "levara"))
            results.append(RetrievalResult(
                content=str(content)[:2000],
                score=score,
                source=str(source),
                metadata=item.get("metadata", {}),
            ))
        return results

    @staticmethod
    def _parse_mcp_response(text: str) -> list[RetrievalResult]:
        import json
        results = []
        for line in text.split("\n"):
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    mcp_result = data.get("result", {})
                    for item in mcp_result.get("content", []):
                        if item.get("type") == "text":
                            try:
                                parsed = json.loads(item["text"])
                                if isinstance(parsed, list):
                                    for entry in parsed:
                                        results.append(RetrievalResult(
                                            content=str(entry.get("content", entry))[:2000],
                                            score=float(entry.get("score", 0.5)),
                                            source="levara",
                                        ))
                            except (json.JSONDecodeError, TypeError):
                                if item["text"].strip():
                                    results.append(RetrievalResult(
                                        content=item["text"][:2000],
                                        score=0.5,
                                        source="levara",
                                    ))
                except json.JSONDecodeError:
                    continue
        return results
