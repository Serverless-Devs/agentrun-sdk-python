"""STEP 事件单元测试

测试 STEP_STARTED / STEP_FINISHED 事件在 AG-UI 和 OpenAI 协议中的行为。
"""

import json
import logging

from fastapi.testclient import TestClient
import pytest

from agentrun.server import (
    AgentEvent,
    AgentRequest,
    AgentRunServer,
    AGUIProtocolHandler,
    EventType,
    OpenAIProtocolHandler,
)


class TestAGUIStepEvents:
    """测试 AG-UI 协议中的 STEP 事件"""

    def get_client(self, invoke_agent):
        server = AgentRunServer(invoke_agent=invoke_agent)
        return TestClient(server.as_fastapi_app())

    def parse_sse_events(self, lines):
        """解析 SSE 行，返回事件字典列表"""
        events = []
        for line in lines:
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    events.append(data)
                except json.JSONDecodeError:
                    pass
        return events

    @pytest.mark.asyncio
    async def test_agui_step_started_event(self):
        """STEP_STARTED with stepName should produce native StepStartedEvent in AG-UI"""

        async def invoke_agent(request: AgentRequest):
            yield AgentEvent(
                event=EventType.STEP_STARTED,
                data={"stepName": "reasoning"},
            )
            yield AgentEvent(
                event=EventType.TEXT,
                data={"delta": "thinking..."},
            )

        client = self.get_client(invoke_agent)
        response = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )

        assert response.status_code == 200
        lines = [line async for line in response.aiter_lines() if line]
        events = self.parse_sse_events(lines)

        # Find STEP_STARTED event
        step_started_events = [e for e in events if e.get("type") == "STEP_STARTED"]
        assert len(step_started_events) == 1
        assert step_started_events[0]["stepName"] == "reasoning"

    @pytest.mark.asyncio
    async def test_agui_step_finished_event(self):
        """STEP_FINISHED with stepName should produce native StepFinishedEvent in AG-UI"""

        async def invoke_agent(request: AgentRequest):
            yield AgentEvent(
                event=EventType.STEP_FINISHED,
                data={"stepName": "reasoning"},
            )

        client = self.get_client(invoke_agent)
        response = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )

        assert response.status_code == 200
        lines = [line async for line in response.aiter_lines() if line]
        events = self.parse_sse_events(lines)

        step_finished_events = [e for e in events if e.get("type") == "STEP_FINISHED"]
        assert len(step_finished_events) == 1
        assert step_finished_events[0]["stepName"] == "reasoning"

    @pytest.mark.asyncio
    async def test_agui_step_started_missing_stepname(self, caplog):
        """STEP_STARTED without stepName should fallback to CustomEvent and log warning"""

        async def invoke_agent(request: AgentRequest):
            yield AgentEvent(
                event=EventType.STEP_STARTED,
                data={"some_field": "value"},
            )

        client = self.get_client(invoke_agent)

        with caplog.at_level(logging.WARNING, logger="agentrun.server.agui_protocol"):
            response = client.post(
                "/ag-ui/agent",
                json={"messages": [{"role": "user", "content": "Hi"}]},
            )

            assert response.status_code == 200
            lines = [line async for line in response.aiter_lines() if line]
            events = self.parse_sse_events(lines)

        # Should fallback to CUSTOM event
        custom_events = [e for e in events if e.get("type") == "CUSTOM"]
        assert len(custom_events) == 1
        assert custom_events[0]["name"] == "step_started"
        assert custom_events[0]["value"] == {"some_field": "value"}

        # Should have logged a warning
        assert any(
            "STEP_STARTED event missing 'stepName' field" in record.message
            for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_agui_step_finished_missing_stepname(self, caplog):
        """STEP_FINISHED without stepName should fallback to CustomEvent and log warning"""

        async def invoke_agent(request: AgentRequest):
            yield AgentEvent(
                event=EventType.STEP_FINISHED,
                data={},
            )

        client = self.get_client(invoke_agent)

        with caplog.at_level(logging.WARNING, logger="agentrun.server.agui_protocol"):
            response = client.post(
                "/ag-ui/agent",
                json={"messages": [{"role": "user", "content": "Hi"}]},
            )

            assert response.status_code == 200
            lines = [line async for line in response.aiter_lines() if line]
            events = self.parse_sse_events(lines)

        # Should fallback to CUSTOM event
        custom_events = [e for e in events if e.get("type") == "CUSTOM"]
        assert len(custom_events) == 1
        assert custom_events[0]["name"] == "step_finished"

        # Should have logged a warning
        assert any(
            "STEP_FINISHED event missing 'stepName' field" in record.message
            for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_agui_step_with_snake_case_step_name(self):
        """STEP events should also accept snake_case 'step_name' field"""

        async def invoke_agent(request: AgentRequest):
            yield AgentEvent(
                event=EventType.STEP_STARTED,
                data={"step_name": "final_answer"},
            )
            yield AgentEvent(
                event=EventType.STEP_FINISHED,
                data={"step_name": "final_answer"},
            )

        client = self.get_client(invoke_agent)
        response = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )

        assert response.status_code == 200
        lines = [line async for line in response.aiter_lines() if line]
        events = self.parse_sse_events(lines)

        step_started = [e for e in events if e.get("type") == "STEP_STARTED"]
        step_finished = [e for e in events if e.get("type") == "STEP_FINISHED"]
        assert len(step_started) == 1
        assert step_started[0]["stepName"] == "final_answer"
        assert len(step_finished) == 1
        assert step_finished[0]["stepName"] == "final_answer"

    @pytest.mark.asyncio
    async def test_agui_full_step_lifecycle(self):
        """Full step lifecycle: reasoning → final_answer in AG-UI"""

        async def invoke_agent(request: AgentRequest):
            # Step 1: reasoning
            yield AgentEvent(
                event=EventType.STEP_STARTED,
                data={"stepName": "reasoning"},
            )
            yield AgentEvent(
                event=EventType.TEXT,
                data={"delta": "Let me think..."},
            )
            yield AgentEvent(
                event=EventType.STEP_FINISHED,
                data={"stepName": "reasoning"},
            )
            # Step 2: final_answer
            yield AgentEvent(
                event=EventType.STEP_STARTED,
                data={"stepName": "final_answer"},
            )
            yield AgentEvent(
                event=EventType.TEXT,
                data={"delta": "The answer is 42."},
            )
            yield AgentEvent(
                event=EventType.STEP_FINISHED,
                data={"stepName": "final_answer"},
            )

        client = self.get_client(invoke_agent)
        response = client.post(
            "/ag-ui/agent",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )

        assert response.status_code == 200
        lines = [line async for line in response.aiter_lines() if line]
        events = self.parse_sse_events(lines)

        # Extract event types in order
        event_types = [e.get("type") for e in events]

        # Verify the lifecycle sequence
        assert "RUN_STARTED" in event_types
        assert "RUN_FINISHED" in event_types

        # Verify step events exist and are in correct order
        step_events = [
            e for e in events if e.get("type") in ("STEP_STARTED", "STEP_FINISHED")
        ]
        assert len(step_events) == 4

        assert step_events[0]["type"] == "STEP_STARTED"
        assert step_events[0]["stepName"] == "reasoning"
        assert step_events[1]["type"] == "STEP_FINISHED"
        assert step_events[1]["stepName"] == "reasoning"
        assert step_events[2]["type"] == "STEP_STARTED"
        assert step_events[2]["stepName"] == "final_answer"
        assert step_events[3]["type"] == "STEP_FINISHED"
        assert step_events[3]["stepName"] == "final_answer"

        # Verify text content is also present
        text_events = [e for e in events if e.get("type") == "TEXT_MESSAGE_CONTENT"]
        assert len(text_events) == 2
        deltas = [e["delta"] for e in text_events]
        assert "Let me think..." in deltas
        assert "The answer is 42." in deltas


class TestOpenAIStepEvents:
    """测试 OpenAI 协议中的 STEP 事件（应被静默跳过）"""

    def get_client(self, invoke_agent):
        server = AgentRunServer(invoke_agent=invoke_agent)
        return TestClient(server.as_fastapi_app())

    @pytest.mark.asyncio
    async def test_openai_skips_step_events(self):
        """STEP events should be silently skipped in OpenAI protocol"""

        async def invoke_agent(request: AgentRequest):
            yield AgentEvent(
                event=EventType.STEP_STARTED,
                data={"stepName": "reasoning"},
            )
            yield AgentEvent(
                event=EventType.TEXT,
                data={"delta": "Hello world"},
            )
            yield AgentEvent(
                event=EventType.STEP_FINISHED,
                data={"stepName": "reasoning"},
            )

        client = self.get_client(invoke_agent)
        response = client.post(
            "/openai/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

        assert response.status_code == 200
        lines = [line async for line in response.aiter_lines() if line]

        # Collect all SSE data
        all_text = "\n".join(lines)

        # Should NOT contain any STEP references
        assert "STEP_STARTED" not in all_text
        assert "STEP_FINISHED" not in all_text
        assert "step_started" not in all_text
        assert "step_finished" not in all_text

        # But SHOULD contain the text content
        found_content = False
        for line in lines:
            if line.startswith("data: ") and not line.startswith("data: [DONE]"):
                try:
                    data = json.loads(line[6:])
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    if delta.get("content") == "Hello world":
                        found_content = True
                except (json.JSONDecodeError, IndexError):
                    pass

        assert found_content, "Text content should be present in OpenAI SSE output"

    @pytest.mark.asyncio
    async def test_openai_skips_step_without_stepname(self):
        """STEP events without stepName should also be skipped in OpenAI protocol"""

        async def invoke_agent(request: AgentRequest):
            yield AgentEvent(
                event=EventType.STEP_STARTED,
                data={},  # no stepName
            )
            yield AgentEvent(
                event=EventType.TEXT,
                data={"delta": "response"},
            )
            yield AgentEvent(
                event=EventType.STEP_FINISHED,
                data={},  # no stepName
            )

        client = self.get_client(invoke_agent)
        response = client.post(
            "/openai/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

        assert response.status_code == 200
        lines = [line async for line in response.aiter_lines() if line]

        all_text = "\n".join(lines)
        assert "STEP" not in all_text
        assert "step_started" not in all_text
        assert "step_finished" not in all_text
