"""Test action runner."""
import pytest
from unittest.mock import AsyncMock, patch

from realtime_demo.dialogue.action_runner import ActionRunner


class TestApiCall:
    @pytest.mark.asyncio
    async def test_api_call_success(self):
        runner = ActionRunner()
        state = {"slots": {}, "variables": {}}
        action = {
            "type": "api_call",
            "method": "POST",
            "url": "https://example.com/api/block",
            "body": {"card": "1234"},
            "timeout_ms": 5000,
            "result_var": "block_result",
            "on_error": "n_err",
        }
        with patch("httpx.AsyncClient.request", new_callable=AsyncMock) as mock_req:
            mock_resp = AsyncMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"success": True}
            mock_req.return_value = mock_resp
            result = await runner.execute_api_call(action, state)
        assert result["success"] is True
        assert state["variables"]["block_result"] == {"success": True}

    @pytest.mark.asyncio
    async def test_api_call_failure_returns_none(self):
        runner = ActionRunner()
        state = {"slots": {}, "variables": {}}
        action = {
            "type": "api_call",
            "method": "POST",
            "url": "https://example.com/api/block",
            "body": {},
            "timeout_ms": 100,
            "result_var": "result",
            "on_error": "n_err",
        }
        with patch("httpx.AsyncClient.request", new_callable=AsyncMock, side_effect=Exception("timeout")):
            result = await runner.execute_api_call(action, state)
        assert result is None


class TestExecute:
    @pytest.mark.asyncio
    async def test_end_action(self):
        runner = ActionRunner()
        action = {"type": "end", "disposition": "resolved"}
        result = await runner.execute(action, state={})
        assert result["completed"] is True

    @pytest.mark.asyncio
    async def test_transfer_action(self):
        runner = ActionRunner()
        action = {"type": "transfer", "reason": "slot_fail", "slots": {"card": "1234"}}
        result = await runner.execute(action, state={})
        assert result["transfer"] is True
        assert result["reason"] == "slot_fail"
