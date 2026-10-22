"""MCP failures must remain actionable without exposing auth or upstream data."""

import unittest
from unittest.mock import AsyncMock, patch

import httpx

from falloutassist_next import rovo, rovo_a2a, rovo_history


class RovoMCPErrorTests(unittest.TestCase):
    def http_error(self, status):
        request = httpx.Request("POST", rovo.MCP_URL,
                                headers={"Authorization": "Basic private-credential"})
        response = httpx.Response(status, request=request, text="private-upstream-body")
        return httpx.HTTPStatusError("private-credential", request=request, response=response)

    def test_rejected_credentials_reach_preview_as_safe_401_diagnostic(self):
        failing_search = AsyncMock(side_effect=self.http_error(401))
        with patch.object(rovo_history, "_search", failing_search):
            with self.assertRaises(rovo_a2a.A2AError) as raised:
                rovo_a2a._historical_keys({"key": "SAC-1", "query_text": "failure"}, [])
        message = str(raised.exception)
        self.assertIn("HTTP 401", message)
        self.assertIn("MCP V2 API token", message)
        self.assertNotIn("private-credential", message)
        self.assertNotIn("private-upstream-body", message)

    def test_nested_mcp_transport_error_preserves_forbidden_status(self):
        error = ExceptionGroup("transport", [ExceptionGroup("session", [self.http_error(403)])])
        message = str(rovo._search_error(error))
        self.assertIn("HTTP 403", message)
        self.assertIn("search:rovo:agent-interface", message)
        self.assertNotIn("private-credential", message)

    def test_network_failures_do_not_claim_invalid_credentials(self):
        for error in (httpx.ConnectError("private-upstream-body"),
                      httpx.ReadTimeout("private-upstream-body")):
            with self.subTest(error=type(error).__name__):
                message = str(rovo._search_error(error))
                self.assertNotIn("401", message)
                self.assertNotIn("private-upstream-body", message)
        self.assertIn("timed out", str(rovo._search_error(httpx.ReadTimeout("timeout"))))

    def test_missing_mcp_token_names_the_correct_connection_field(self):
        message = str(rovo._search_error(RuntimeError("ROVO_API_TOKEN is required")))
        self.assertIn("MCP V2 API token", message)
        self.assertIn("Settings > Connections", message)

    def test_session_cleanup_preserves_safe_tool_diagnostic(self):
        error = rovo.RovoMCPError("Semantic search requires search:rovo:agent-interface.")
        wrapped = ExceptionGroup("session closed", [error])
        self.assertIs(rovo._search_error(wrapped), error)


if __name__ == "__main__":
    unittest.main()
