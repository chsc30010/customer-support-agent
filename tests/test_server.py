import pytest
from fastapi.testclient import TestClient

from support_agent.config import Settings
from support_agent.server import create_app
from support_agent.telephony import signature

TOKEN = "test-auth-token"
BASE = "https://kestrel.example.test"


def settings(**kwargs):
    base = dict(
        twilio_auth_token=TOKEN,
        twilio_account_sid="AC0",
        public_base_url=BASE,
        handoff_phone_number="+15550000000",
    )
    base.update(kwargs)
    return Settings(**base)


@pytest.fixture
def client():
    return TestClient(create_app(settings=settings()))


def post_signed(client, path, params, config=None):
    config = config or settings()
    url = signature.canonical_url(config.public_base_url, path)
    header = signature.sign_for_testing(config.twilio_auth_token, url, params)
    return client.post(path, data=params, headers={"X-Twilio-Signature": header})


def test_health_reports_what_is_wired_up(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["kb_passages"] > 40
    assert body["webhook_verification"] is True


def test_an_unsigned_webhook_is_refused(client):
    response = client.post("/twilio/voice", data={"CallSid": "CA1"})
    assert response.status_code == 403


def test_a_webhook_signed_with_the_wrong_token_is_refused(client):
    url = signature.canonical_url(BASE, "/twilio/voice")
    header = signature.sign_for_testing("not-the-token", url, {"CallSid": "CA1"})
    response = client.post(
        "/twilio/voice", data={"CallSid": "CA1"}, headers={"X-Twilio-Signature": header}
    )
    assert response.status_code == 403


def test_signature_covers_the_parameters_not_just_the_url(client):
    url = signature.canonical_url(BASE, "/twilio/voice")
    header = signature.sign_for_testing(TOKEN, url, {"CallSid": "CA1"})
    tampered = {"CallSid": "CA1", "From": "+15551234567"}
    response = client.post(
        "/twilio/voice", data=tampered, headers={"X-Twilio-Signature": header}
    )
    assert response.status_code == 403


def test_without_credentials_webhooks_fail_closed():
    unconfigured = TestClient(create_app(settings=Settings()))
    response = unconfigured.post("/twilio/voice", data={"CallSid": "CA1"})
    assert response.status_code == 503
    assert "verification is not configured" in response.json()["detail"]


def test_unsigned_webhooks_can_be_allowed_for_local_testing():
    local = TestClient(create_app(settings=Settings(allow_unsigned_webhooks=True)))
    assert local.post("/twilio/voice", data={"CallSid": "CA1"}).status_code == 200


def test_the_call_opens_with_a_greeting_inside_a_gather(client):
    body = post_signed(client, "/twilio/voice", {"CallSid": "CA1"}).text
    assert body.startswith('<?xml version="1.0" encoding="UTF-8"?><Response>')
    assert "<Gather" in body and "<Say" in body
    assert body.index("<Gather") < body.index("<Say")


def test_an_answered_turn_keeps_listening(client):
    body = post_signed(
        client,
        "/twilio/voice/turn",
        {"CallSid": "CA2", "SpeechResult": "how do I get a return label", "Confidence": "0.94"},
    ).text
    assert "prepaid" in body
    assert "<Gather" in body
    assert "<Dial>" not in body


def test_pressing_zero_transfers_the_call(client):
    body = post_signed(client, "/twilio/voice/turn", {"CallSid": "CA3", "Digits": "0"}).text
    assert "<Dial" in body and "+15550000000" in body
    assert "<Gather" not in body


def test_without_a_transfer_number_the_caller_is_queued():
    queued = TestClient(create_app(settings=settings(handoff_phone_number="")))
    body = post_signed(
        queued,
        "/twilio/voice/turn",
        {"CallSid": "CA4", "SpeechResult": "I want to file a formal complaint"},
        settings(handoff_phone_number=""),
    ).text
    assert "<Enqueue>escalations</Enqueue>" in body


def test_silence_re_prompts_then_hangs_up(client):
    first = post_signed(client, "/twilio/voice/turn", {"CallSid": "CA5"}).text
    assert "did not catch that" in first
    assert "empty=1" in first

    response = client.post(
        "/twilio/voice/turn?empty=2",
        data={"CallSid": "CA5"},
        headers={
            "X-Twilio-Signature": signature.sign_for_testing(
                TOKEN,
                signature.canonical_url(BASE, "/twilio/voice/turn", "empty=2"),
                {"CallSid": "CA5"},
            )
        },
    )
    assert "<Hangup/>" in response.text


def test_sms_replies_with_a_message(client):
    body = post_signed(
        client,
        "/twilio/sms",
        {"From": "+15557654321", "Body": "how much is the family plan", "MessageSid": "SM1"},
    ).text
    assert "<Message>" in body
    assert "five dollars" in body


def test_chat_returns_json_with_the_decision(client):
    body = client.post(
        "/chat", json={"conversation_id": "c1", "text": "I forgot my password"}
    ).json()
    assert body["intent"] == "account_access"
    assert body["escalation"]["escalate"] is False
    assert body["citations"][0]["title"] == "Trouble signing in"


def test_email_gets_a_subject_line(client):
    body = client.post(
        "/email",
        json={"from": "a@example.test", "subject": "Return", "body": "how do I return the doorbell"},
    ).json()
    assert body["subject"].startswith("Re: ")
