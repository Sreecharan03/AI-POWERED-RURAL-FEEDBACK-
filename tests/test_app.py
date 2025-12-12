import sys
import types

import pytest
import httpx

# Provide lightweight stubs for optional heavy deps if missing (e.g., asyncpg, database)
if "asyncpg" not in sys.modules:
    async def _fake_create_pool(*args, **kwargs):
        return None
    sys.modules["asyncpg"] = types.SimpleNamespace(create_pool=_fake_create_pool)

if "database" not in sys.modules:
    async def _fake_check_database_health():
        return {"overall_healthy": True}
    sys.modules["database"] = types.SimpleNamespace(
        db_manager=None,
        db_ops=None,
        check_database_health=_fake_check_database_health,
    )

if "speech_services" not in sys.modules:
    async def _fake_process_voice_to_text(*args, **kwargs):
        return {"success": True, "best_transcript": "voice text", "best_confidence": 0.9}
    async def _fake_process_text_to_voice(*args, **kwargs):
        return {"success": True, "audio_content": b"audio-bytes"}
    async def _fake_check_speech_services_health():
        return {"speech_client": True, "tts_client": True, "credentials": True}
    sys.modules["speech_services"] = types.SimpleNamespace(
        speech_processor=None,
        process_voice_to_text=_fake_process_voice_to_text,
        process_text_to_voice=_fake_process_text_to_voice,
        check_speech_services_health=_fake_check_speech_services_health,
    )

if "conversation_engine" not in sys.modules:
    async def _stub_start_conversation(conv_id: str):
        return {
            "session_started": True,
            "initial_response": {
                "telugu_response": "hello",
                "ai_model": "stub",
                "confidence_score": 0.9,
            },
        }
    async def _stub_process_message(conv_id: str, user_input: str, audio_metadata=None):
        return {
            "success": True,
            "ai_response": {
                "telugu_response": f"echo: {user_input}",
                "confidence_score": 0.9,
                "ai_model": "stub",
            },
            "conversation_state": {"conversation_id": conv_id, "turns": 1},
        }
    async def _stub_end_conversation_session(conv_id: str, skip_save: bool = False):
        return {"success": True, "summary": {"ai_insights": {"done": True}}}
    sys.modules["conversation_engine"] = types.SimpleNamespace(
        jan_spandana_ai=None,
        start_conversation=_stub_start_conversation,
        process_message=_stub_process_message,
        end_conversation_session=_stub_end_conversation_session,
    )

if "data_processor" not in sys.modules:
    sys.modules["data_processor"] = types.SimpleNamespace(ai_conversation_processor=None)

import main
from main import app, conversation_store, ai_monitor

# Run all tests with the anyio plugin (async support)
pytestmark = pytest.mark.anyio


@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    async def fake_db_health():
        return {"overall_healthy": True}

    async def fake_speech_health():
        return {"speech_client": True, "tts_client": True, "credentials": True}

    async def fake_ai_health():
        return (
            {
                "conversation_engine": True,
                "data_processor": True,
                "gemini_api": True,
                "redis_cache": True,
                "ai_performance": "operational",
            },
            {"ai_success_rate": 1.0, "total_ai_calls": 0, "avg_ai_response_time": 10},
        )

    async def noop_init_redis():
        return None

    async def noop_close_redis():
        return None

    async def fake_start_conversation(conv_id: str):
        return {
            "session_started": True,
            "initial_response": {
                "telugu_response": "hello",
                "ai_model": "stub",
                "confidence_score": 0.9,
            },
        }

    async def fake_process_message(conv_id: str, user_input: str, audio_metadata=None):
        return {
            "success": True,
            "ai_response": {
                "telugu_response": f"echo: {user_input}",
                "confidence_score": 0.9,
                "ai_model": "stub",
            },
            "conversation_state": {"conversation_id": conv_id, "turns": 1},
        }

    async def fake_end_conversation(conv_id: str, skip_save: bool = False):
        return {"success": True, "summary": {"ai_insights": {"done": True}}}

    async def fake_process_voice_to_text(content: bytes, fmt: str):
        return {"success": True, "best_transcript": "voice text", "best_confidence": 0.88}

    async def fake_process_text_to_voice(text: str, voice: str):
        return {"success": True, "audio_content": b"audio-bytes"}

    monkeypatch.setattr(main, "check_database_health", fake_db_health)
    monkeypatch.setattr(main, "check_speech_services_health", fake_speech_health)
    monkeypatch.setattr(ai_monitor, "check_ai_services_health", fake_ai_health)
    monkeypatch.setattr(main, "init_redis", noop_init_redis)
    monkeypatch.setattr(main, "close_redis", noop_close_redis)
    monkeypatch.setattr(main, "start_conversation", fake_start_conversation)
    monkeypatch.setattr(main, "process_message", fake_process_message)
    monkeypatch.setattr(main, "end_conversation_session", fake_end_conversation)
    monkeypatch.setattr(main, "process_voice_to_text", fake_process_voice_to_text)
    monkeypatch.setattr(main, "process_text_to_voice", fake_process_text_to_voice)

    ai_monitor.ai_performance_history.clear()
    conversation_store.local_store.clear()
    yield


def get_test_client():
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def test_health_is_healthy():
    async with get_test_client() as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "healthy"
    assert payload["services"]["database"] is True
    assert payload["services"]["cache"] is True


async def test_conversation_text_flow():
    async with get_test_client() as client:
        start_resp = await client.post("/api/v1/conversation/start", json={})
        assert start_resp.status_code == 200
        conv_id = start_resp.json()["conversation_id"]

        text_resp = await client.post(
            "/api/v1/conversation/text",
            json={"conversation_id": conv_id, "user_input": "hi there"},
        )
        assert text_resp.status_code == 200
        text_payload = text_resp.json()
        assert text_payload["success"] is True
        assert text_payload["ai_response_text"].startswith("echo: hi there")

        end_resp = await client.post(
            "/api/v1/conversation/end",
            json={"conversation_id": conv_id, "completion_reason": "test_done"},
        )
        assert end_resp.status_code == 200
        assert end_resp.json()["conversation_ended"] is True
