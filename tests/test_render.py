from support_agent.models import AgentReply, Answer, Channel, Classification, Intent, Passage
from support_agent.render import render, segment
from support_agent.render.voice import VoiceRenderer

PASSAGE = Passage(
    article_id="returns", article_title="Returns and refunds",
    section="Starting a return", text="body", url="https://help.example.test/returns",
)


def reply(text, channel=Channel.CHAT, grounded=True):
    return AgentReply(
        conversation_id="t",
        channel=channel,
        text=text,
        classification=Classification(intent=Intent.RETURNS_REFUND, confidence=0.9),
        answer=Answer(text=text, citations=[PASSAGE], grounded=grounded),
    )


def test_voice_never_reads_a_url_aloud():
    out = render(reply("See https://help.example.test/returns for details.", Channel.VOICE))
    assert "https" not in out.ssml
    assert "text you the link" in out.text


def test_voice_spells_out_a_reference_the_caller_must_write_down():
    out = render(reply("Your return reference is KH-482913.", Channel.VOICE))
    assert 'interpret-as="characters"' in out.ssml
    assert "KH-482913" in out.ssml


def test_voice_pauses_between_sentences():
    out = render(reply("First sentence here. Second sentence here.", Channel.VOICE))
    assert out.ssml.count('<break time="350ms"/>') == 1
    assert out.ssml.startswith("<speak>") and out.ssml.endswith("</speak>")


def test_voice_adds_pauses_inside_a_long_sentence():
    long_one = (
        "Open the Kestrel app, go to Account, then Orders, choose Return next to "
        "the item, and we will email you a prepaid label straight away."
    )
    out = render(reply(long_one, Channel.VOICE))
    assert '<break time="200ms"/>' in out.ssml


def test_voice_does_not_rewrite_what_the_customer_is_told():
    original = "Hold the button for 15 seconds, then set the camera up again."
    out = render(reply(original, Channel.VOICE))
    assert out.text == original


def test_voice_escapes_xml():
    out = VoiceRenderer().render(reply("Fees & charges <are> listed.", Channel.VOICE))
    assert "&amp;" in out.ssml and "&lt;are&gt;" in out.ssml


def test_voice_strips_markdown():
    out = render(reply("Go to **Account**, then `Orders`.", Channel.VOICE))
    assert "*" not in out.text and "`" not in out.text


def test_a_short_sms_is_one_unnumbered_message():
    parts = segment("Your refund takes three to five business days.")
    assert len(parts) == 1
    assert "(1/1)" not in parts[0]


def test_a_long_sms_is_split_and_numbered():
    parts = segment("word " * 120)
    assert len(parts) > 1
    assert parts[0].endswith("(1/{})".format(len(parts)))
    assert all(len(p) <= 160 for p in parts)


def test_sms_never_runs_past_three_segments():
    parts = segment("word " * 600)
    assert len(parts) == 3
    assert parts[-1].endswith("...(3/3)") or "..." in parts[-1]


def test_chat_gets_a_clickable_citation():
    out = render(reply("Here is how."))
    assert "[Returns and refunds - Starting a return](https://help.example.test/returns)" in out.text


def test_an_ungrounded_reply_cites_nothing():
    out = render(reply("I could not find that.", grounded=False))
    assert "https" not in out.text


def test_email_gets_a_subject_and_a_sign_off():
    out = render(reply("Here is how.", Channel.EMAIL))
    assert out.subject == "Re: Your return"
    assert out.text.endswith("Kestrel Home Support")
