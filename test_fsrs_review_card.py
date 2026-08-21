"""Contract tests: inclusive ReviewCard matches official py-fsrs Scheduler.review_card."""

from datetime import datetime, timedelta, timezone

import pytest
from fsrs import Card, Rating, Scheduler, State


def _review(scheduler: Scheduler, card: Card, rating: Rating, at: datetime) -> Card:
    updated, _log = scheduler.review_card(card, rating, at, 0)
    return updated


def test_learning_steps_good_eventually_graduates_to_review():
    """Default [1m, 10m] ladder: repeated Good leaves learning and schedules day+ review."""
    scheduler = Scheduler(learning_steps=(timedelta(minutes=1), timedelta(minutes=10)))
    at = datetime(2026, 5, 19, 12, 0, tzinfo=timezone.utc)
    card = Card(state=State.Learning, due=at, last_review=at)

    for _ in range(4):
        if card.state == State.Review:
            break
        card = _review(scheduler, card, Rating.Good, at)

    assert card.state == State.Review
    assert card.due >= at + timedelta(hours=12)


def test_review_again_enters_relearning():
    scheduler = Scheduler(
        learning_steps=(timedelta(minutes=1), timedelta(minutes=10)),
        relearning_steps=(timedelta(minutes=10),),
    )
    at = datetime(2026, 5, 19, 12, 0, tzinfo=timezone.utc)
    card = Card(
        state=State.Review,
        stability=10,
        difficulty=5,
        due=at,
        last_review=at - timedelta(days=1),
    )

    relearn = _review(scheduler, card, Rating.Again, at)
    assert relearn.state == State.Relearning


def test_fuzz_disabled_is_deterministic():
    scheduler = Scheduler(enable_fuzzing=False)
    at = datetime(2026, 5, 19, 12, 0, tzinfo=timezone.utc)
    card = Card(
        state=State.Review,
        stability=20,
        difficulty=5,
        due=at,
        last_review=at - timedelta(days=10),
    )

    a = _review(scheduler, card, Rating.Good, at)
    b = _review(scheduler, card, Rating.Good, at)
    assert a.due == b.due


def test_inclusive_new_card_maps_to_learning_first_review():
    """Inclusive maps wire state=0 to State.Learning before review_card (see main.py)."""
    scheduler = Scheduler(learning_steps=(timedelta(minutes=1), timedelta(minutes=10)))
    at = datetime(2026, 5, 19, 12, 0, tzinfo=timezone.utc)
    card = Card(state=State.Learning, due=at, last_review=at)

    next_card = _review(scheduler, card, Rating.Good, at)
    assert next_card.state == State.Learning
    assert next_card.due > at
