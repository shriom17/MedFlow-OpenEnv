"""Tests for composable reward rubric scoring."""

from app.tasks import default_rubric


def test_default_rubric_prefers_good_state():
    rubric = default_rubric()

    good_state = {
        "emergency_wait_minutes": [0.0],
        "specialization_match": True,
        "bed_utilization": 0.7,
    }
    bad_state = {
        "emergency_wait_minutes": [12.0],
        "specialization_match": False,
        "bed_utilization": 0.1,
    }

    assert rubric.score(good_state) > rubric.score(bad_state)


def test_default_rubric_accepts_scalar_wait():
    rubric = default_rubric()

    score = rubric.score({
        "emergency_wait_minutes": 5,
        "specialization_match": True,
        "bed_utilization": 0.5,
    })

    assert isinstance(score, float)