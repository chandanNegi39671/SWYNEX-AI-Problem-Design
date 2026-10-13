import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pytest
from predict import route, load
B = load()

def test_known_intent_auto_routed():
    r = route("I still have not received my card, when will it arrive?", B)
    assert r["intent"].startswith("card_arrival") or r["intent"] == "card_delivery_estimate"

def test_out_of_scope_goes_to_human():
    assert route("What's the weather like in Delhi tomorrow?", B)["decision"] == "HUMAN_REVIEW"

def test_output_schema():
    r = route("how do I change my pin", B)
    assert set(r) == {"decision", "intent", "confidence", "top_k"} and len(r["top_k"]) == 3
    assert 0 <= r["confidence"] <= 1

def test_typo_robustness():
    r = route("how do i chnage my pin", B)
    assert "change_pin" in [c for c, _ in r["top_k"]]

def test_empty_input_does_not_crash():
    assert route("", B)["decision"] in {"AUTO_ROUTE", "HUMAN_REVIEW"}
