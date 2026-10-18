"""The 7-day testing-mode refresh token deadline, before a cron finds out.

Google issues refresh tokens that expire after 7 days when the OAuth consent
screen is left in "Testing", and non-expiring ones once the app is published.
Nothing in the token file records that deadline, so a daily cron would just
stop working a week after setup with no other signal.
"""

from __future__ import annotations

import os
import time

import pytest

from ytauto.config import Settings
from ytauto.youtube import auth as ytauth


@pytest.fixture
def token(tmp_path, monkeypatch):
    """A token.json whose age the test controls, wired into get_settings()."""
    p = tmp_path / "token.json"
    p.write_text("{}")

    def _settings():
        return Settings(token_path=p)

    monkeypatch.setattr(ytauth, "get_settings", _settings)
    return p


def _age_days(p, days):
    when = time.time() - days * 86400
    os.utime(p, (when, when))


def test_a_fresh_token_is_not_flagged(token):
    assert ytauth.token_age_days() < 1.0
    assert ytauth.auth_status()["reauth_due"] is False


def test_no_token_means_no_age_rather_than_zero(tmp_path, monkeypatch):
    """A missing file is a different problem from an old one."""
    monkeypatch.setattr(
        ytauth, "get_settings",
        lambda: Settings(token_path=tmp_path / "absent.json"),
    )
    assert ytauth.token_age_days() is None
    assert ytauth.auth_status()["reauth_due"] is False


def test_a_token_older_than_six_days_is_flagged_for_reauth(token):
    _age_days(token, 6.5)
    status = ytauth.auth_status()
    assert status["reauth_due"] is True
    assert status["token_age_days"] > 6
    assert status["testing_token_days"] == 7.0


def test_the_warn_threshold_lands_inside_the_google_deadline():
    """Warn at 6 days, expire at 7: the warning has to arrive first."""
    assert ytauth.TESTING_REFRESH_TOKEN_DAYS == 7.0
    assert ytauth._REAUTH_WARN_DAYS < ytauth.TESTING_REFRESH_TOKEN_DAYS


def test_the_documented_deadline_matches_google_policy():
    """Google: testing-mode refresh tokens expire 7 days after consent."""
    assert ytauth.TESTING_REFRESH_TOKEN_DAYS == 7.0
