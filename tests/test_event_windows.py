"""Tests for event window configuration and table generation."""

from stressnet.config import load_events


def test_events_load():
    events = load_events()
    assert isinstance(events, dict)
    assert len(events) >= 5


def test_usdc_svb_is_primary():
    events = load_events()
    assert "usdc_svb_2023" in events
    assert events["usdc_svb_2023"]["primary"] is True


def test_all_events_have_analysis_windows():
    events = load_events()
    for event_id, cfg in events.items():
        w = cfg.get("analysis_window_utc", [])
        assert len(w) == 2, f"Event {event_id} missing analysis_window_utc"
        assert w[0] < w[1], f"Event {event_id} has inverted window"


def test_all_events_have_mechanism():
    events = load_events()
    for event_id, cfg in events.items():
        assert cfg.get("mechanism"), f"Event {event_id} missing mechanism"
