"""Pronunciation fixes apply to whole words only."""

from support_agent.models import AgentReply, Channel, Classification, Intent
from support_agent.render import render
from support_agent.render.voice import _despell


def spoken(text):
    reply = AgentReply(
        conversation_id="t",
        channel=Channel.VOICE,
        text=text,
        classification=Classification(intent=Intent.PRODUCT_INFO, confidence=0.9),
    )
    return render(reply).text


def test_words_that_merely_contain_a_key_are_left_alone():
    # These used to come out as "acti V A T e", "pri V A T e" and "1two K".
    for text in [
        "You can activate night vision under Settings.",
        "Your recordings are private to your account.",
        "Clips are stored at 12K resolution.",
    ]:
        assert _despell(text) == text, text


def test_whole_word_keys_are_still_converted():
    assert (
        _despell("We can reissue it with your VAT number.")
        == "We can reissue it with your V A T number."
    )
    assert "two point four gigahertz" in _despell("Cameras need a 2.4GHz network.")
    assert _despell("Agents work 8am to 8pm.") == "Agents work 8 a m to 8 p m."
    assert "wi-fi" in _despell("Check your wifi signal.")


def test_a_key_inside_a_longer_number_is_left_alone():
    # "5GHz" must not match inside "2.5GHz".
    assert _despell("It also supports 2.5GHz.") == "It also supports 2.5GHz."


def test_the_voice_renderer_speaks_whole_words_only():
    out = spoken("You can activate night vision, and we can add your VAT number.")
    assert "activate" in out
    assert "V A T number" in out
