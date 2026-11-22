from refusal_degeneracy.judge import is_refusal, refusal_rate


def test_is_refusal_detects_phrases():
    assert is_refusal("I'm sorry, I cannot help with that.")
    assert is_refusal("As an AI language model, I won't.")


def test_is_refusal_false_on_compliance():
    assert not is_refusal("Sure, here are the steps you asked for.")


def test_refusal_rate():
    assert refusal_rate(["I cannot", "sure thing", "I'm sorry"]) == 2 / 3
    assert refusal_rate([]) == 0.0
