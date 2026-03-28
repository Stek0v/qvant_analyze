"""Async adapter для Cognee MCP (SSE). ARCH-07.

Cognee использует MCP протокол через SSE (Server-Sent Events).
Endpoint: POST /mcp, Accept: application/json, text/event-stream.
Требует initialize → tools/call для каждой сессии.
"""

from __future__ import annotations

import json
import logging

import httpx

from router_api.models import RetrievalResult

logger = logging.getLogger(__name__)


class CogneeAdapter:
    def __init__(self, base_url: str = "http://127.0.0.1:9000", timeout: float = 30.0):
        self._base_url = base_url
        self._mcp_url = f"{base_url}/mcp"
        self._timeout = timeout
        self._headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }

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
        search_type: str = "GRAPH_COMPLETION",
    ) -> list[RetrievalResult]:
        """Поиск через Cognee MCP.

        Args:
            query: Поисковый запрос
            mode: "shallow" (top_k=5) или "deep" (top_k=20)
            search_type: Тип поиска Cognee (GRAPH_COMPLETION, HYBRID и т.д.)
        """
        top_k = 5 if mode == "shallow" else 20

        try:
            session_id = await self._initialize()
            if not session_id:
                logger.warning("Cognee: failed to initialize MCP session")
                return []

            result = await self._call_tool(
                session_id=session_id,
                tool_name="search",
                arguments={
                    "search_query": query,
                    "search_type": search_type,
                    "top_k": top_k,
                },
            )

            return self._parse_results(result)

        except Exception as e:
            logger.warning(f"Cognee search failed: {e}")
            return []

    async def _initialize(self) -> str | None:
        """MCP initialize handshake, возвращает session_id."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            r = await client.post(self._mcp_url, headers=self._headers, json={
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "qvant-router", "version": "0.1"},
                },
            })
            return r.headers.get("mcp-session-id")

    async def _call_tool(
        self, session_id: str, tool_name: str, arguments: dict
    ) -> dict:
        """Вызов MCP tool и парсинг SSE ответа."""
        headers = {**self._headers, "Mcp-Session-Id": session_id}

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            r = await client.post(self._mcp_url, headers=headers, json={
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            })

            # Parse SSE response
            for line in r.text.split("\n"):
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    if "result" in data:
                        return data["result"]
                    if "error" in data:
                        logger.warning(f"Cognee MCP error: {data['error']}")
                        return {}
        return {}

    @staticmethod
    def _parse_results(result: dict) -> list[RetrievalResult]:
        """Преобразовать Cognee MCP ответ в список RetrievalResult."""
        results = []

        content_list = result.get("content", [])
        for item in content_list:
            if item.get("type") == "text":
                text = item.get("text", "")
                # Cognee может вернуть JSON строку с результатами
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, list):
                        for entry in parsed:
                            results.append(RetrievalResult(
                                content=entry.get("text", entry.get("content", str(entry))),
                                score=float(entry.get("score", entry.get("relevance", 0.5))),
                                source=entry.get("source", entry.get("id", "")),
                                metadata=entry if isinstance(entry, dict) else {},
                            ))
                    elif isinstance(parsed, dict):
                        results.append(RetrievalResult(
                            content=parsed.get("text", str(parsed)),
                            score=float(parsed.get("score", 0.5)),
                            source=parsed.get("source", ""),
                            metadata=parsed,
                        ))
                except (json.JSONDecodeError, TypeError):
                    # Plaintext result
                    if text.strip():
                        results.append(RetrievalResult(
                            content=text, score=0.5, source="cognee",
                        ))

        return results
