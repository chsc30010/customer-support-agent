from support_agent.classify.heuristic import HeuristicClassifier, normalize, score_sentiment
from support_agent.models import Channel, Conversation, InboundMessage, Intent, Sentiment

classifier = HeuristicClassifier()


def message(text, **kwargs):
    return InboundMessage(conversation_id="t", channel=Channel.CHAT, text=text, **kwargs)


def test_normalize_folds_apostrophes_and_negations():
    assert normalize("I can't log in!!") == "i cant log in"
    assert normalize("I cannot log in") == "i cant log in"
    assert normalize("It will not connect") == "it wont connect"


def test_intents_are_recognised():
    pairs = [
        ("where is my order", Intent.ORDER_STATUS),
        ("how do I get a return label", Intent.RETURNS_REFUND),
        ("I was charged twice", Intent.BILLING),
        ("I forgot my password", Intent.ACCOUNT_ACCESS),
        ("my camera keeps going offline", Intent.TECHNICAL_ISSUE),
        ("I want to cancel my subscription", Intent.CANCELLATION),
        ("is it compatible with google home", Intent.PRODUCT_INFO),
        ("I want to file a formal complaint", Intent.COMPLAINT),
    ]
    for text, expected in pairs:
        assert classifier.classify(message(text)).intent is expected, text


def test_cancelling_an_order_is_not_cancelling_a_subscription():
    assert classifier.classify(message("I need to cancel my order")).intent is Intent.RETURNS_REFUND
    assert classifier.classify(message("I need to cancel my plan")).intent is Intent.CANCELLATION


def test_unrecognised_text_is_unknown_with_zero_confidence():
    result = classifier.classify(message("do you sponsor local football teams"))
    assert result.intent is Intent.UNKNOWN
    assert result.confidence == 0.0


def test_pressing_zero_asks_for_a_human():
    assert classifier.classify(message("", digits="0")).wants_human


def test_asking_for_a_supervisor_asks_for_a_human():
    assert classifier.classify(message("can I speak to a supervisor")).wants_human


def test_sentiment_grades_heat():
    assert score_sentiment("thanks, that worked perfectly")[0] is Sentiment.POSITIVE
    assert score_sentiment("my order is late")[0] is Sentiment.NEUTRAL
    assert score_sentiment("this is the third time and nobody has replied")[0] is Sentiment.FRUSTRATED
    assert score_sentiment("this is absolutely ridiculous")[0] is Sentiment.ANGRY


def test_shouting_counts_towards_anger():
    calm, _ = score_sentiment("still no update and nobody has replied")
    shouted, _ = score_sentiment("STILL NO UPDATE AND NOBODY HAS REPLIED!!")
    assert calm is Sentiment.FRUSTRATED
    assert shouted is Sentiment.ANGRY


def test_poor_speech_recognition_lowers_confidence():
    clear = classifier.classify(message("I forgot my password", speech_confidence=0.95))
    muddy = classifier.classify(message("I forgot my password", speech_confidence=0.3))
    assert muddy.confidence < clear.confidence


def test_short_follow_up_inherits_the_previous_intent():
    conversation = Conversation(id="t", channel=Channel.VOICE)
    conversation.last_intent = Intent.TECHNICAL_ISSUE
    result = classifier.classify(message("yes I already did"), conversation)
    assert result.intent is Intent.TECHNICAL_ISSUE
    assert result.confidence == 0.5


def test_a_long_unrelated_turn_does_not_inherit():
    conversation = Conversation(id="t", channel=Channel.VOICE)
    conversation.last_intent = Intent.TECHNICAL_ISSUE
    result = classifier.classify(
        message("do you sponsor any local football teams in the area at all"), conversation
    )
    assert result.intent is Intent.UNKNOWN
