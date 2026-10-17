"""The detector: three signals, any one escalates, biased to fire."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def model(plugin):
    return plugin.detector.TrigramModel.load()


def test_english_stays_local(plugin, model):
    text = (
        "Hey, just a quick note to say the meeting moved to Thursday afternoon. Can you bring "
        "the spreadsheet and the notes from last week? Thanks, talk soon."
    )
    d = plugin.detector.decide(text, "en", model)
    assert not d.escalate, d


def test_whisper_language_mi_escalates(plugin, model):
    d = plugin.detector.decide("anything at all", "mi", model)
    assert d.escalate and "language" in d.reasons


def test_any_macron_escalates(plugin, model):
    d = plugin.detector.decide("We met at the whānau day on Saturday.", "en", model)
    assert d.escalate and "macron" in d.reasons


def test_mixed_speech_without_macrons_trips_the_trigram(plugin, model):
    # Whisper in English mode drops macrons; the trigram signal is for this.
    text = (
        "So I wanted to start by saying tena koutou katoa, nga mihi nui ki a koutou mo te "
        "tautoko, and then we can go through the agenda for the hui next week."
    )
    d = plugin.detector.decide(text, "en", model)
    assert d.escalate and "trigram" in d.reasons, d


def test_te_reo_without_macrons_scores_high(plugin, model):
    text = "Kia ora koutou katoa. Ko te kaupapa o tenei hui he korero mo nga mahi o te marae."
    assert model.score(text) > 0.5


def test_empty_text_scores_zero(plugin, model):
    assert model.score("") == 0.0
    assert not plugin.detector.decide("", "en", model).escalate


def test_words_keep_macrons(plugin):
    assert plugin.detector.words("Tēnā KOUTOU, kōrero!") == ["tēnā", "koutou", "kōrero"]


def test_model_threshold_is_recorded(plugin, model):
    assert 0.0 < model.threshold < 0.5
