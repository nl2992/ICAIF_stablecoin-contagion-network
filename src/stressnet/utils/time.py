"""Time utilities: nanosecond timestamps, UTC conversion, event-time helpers."""

import time
from datetime import datetime, timezone


def now_ns() -> int:
    """Return current wall-clock time in nanoseconds since Unix epoch."""
    return time.time_ns()


def now_utc() -> datetime:
    """Return current UTC datetime (timezone-aware)."""
    return datetime.now(tz=timezone.utc)


def ns_to_utc(ns: int) -> datetime:
    """Convert a nanosecond Unix timestamp to a UTC datetime."""
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)


def utc_to_ns(dt: datetime) -> int:
    """Convert a UTC datetime to nanoseconds since Unix epoch."""
    return int(dt.timestamp() * 1e9)


def floor_to_second(ns: int) -> int:
    """Floor a nanosecond timestamp to the nearest second boundary."""
    return (ns // 1_000_000_000) * 1_000_000_000


def floor_to_minute(ns: int) -> int:
    """Floor a nanosecond timestamp to the nearest minute boundary."""
    return (ns // 60_000_000_000) * 60_000_000_000


def date_str(ns: int) -> str:
    """Return ISO date string (YYYY-MM-DD) from a nanosecond timestamp."""
    return ns_to_utc(ns).strftime("%Y-%m-%d")


def hour_str(ns: int) -> str:
    """Return zero-padded hour string (HH) from a nanosecond timestamp."""
    return ns_to_utc(ns).strftime("%H")


def event_time_seconds(ts_utc: datetime, onset_utc: datetime) -> float:
    """Return seconds relative to the shock onset (T=0).

    Negative values indicate pre-event time.
    """
    return (ts_utc - onset_utc).total_seconds()


def parse_iso_utc(s: str) -> datetime:
    """Parse an ISO-8601 UTC string to a timezone-aware datetime."""
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
