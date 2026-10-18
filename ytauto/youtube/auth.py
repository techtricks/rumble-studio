"""YouTube OAuth2: one-time browser consent, then silent refresh forever."""

from __future__ import annotations

import json
import os
import stat
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from ..config import get_settings
from ..logging_setup import get_logger

log = get_logger("youtube.auth")

# upload is the only scope strictly required for publishing. Adding
# youtube.readonly lets us auto-detect the channel id on first auth.
SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
]


class AuthError(RuntimeError):
    pass


def _chmod_600(path: Path) -> None:
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:  # pragma: no cover - non-POSIX
        pass


def credentials_path(client_secret: Path) -> Optional[Path]:
    """Return the actual secret file, tolerating Google's
    `client_secret_123.apps.googleusercontent.com.json` naming."""
    if not client_secret.exists():
        matches = sorted(client_secret.parent.glob("client_secret*.json"))
        if not matches:
            return None
        return matches[0]
    return client_secret


def load_credentials(refresh: bool = True) -> Optional[Credentials]:
    s = get_settings()
    if not s.token_path.exists():
        return None
    try:
        creds = Credentials.from_authorized_user_file(str(s.token_path), SCOPES)
    except (ValueError, json.JSONDecodeError) as exc:
        log.error("token file is corrupt (%s); re-run `auth`", exc)
        return None
    if refresh and not creds.valid and creds.expired and creds.refresh_token:
        log.info("refreshing access token")
        creds.refresh(Request())
        save_credentials(creds)
    return creds


def save_credentials(creds: Credentials) -> None:
    s = get_settings()
    s.token_path.parent.mkdir(parents=True, exist_ok=True)
    s.token_path.write_text(creds.to_json(), encoding="utf-8")
    _chmod_600(s.token_path)
    log.info("credentials saved to %s", s.token_path)


def authorize() -> Credentials:
    """Run the interactive consent flow. Opens a browser tab once."""
    s = get_settings()
    secret = credentials_path(s.client_secret_path)
    if secret is None:
        raise AuthError(
            f"No OAuth client found. Expected {s.client_secret_path} "
            "(or client_secret*.json in secrets/). See README 'YouTube API setup'."
        )
    log.info("starting OAuth flow using %s", secret.name)
    # No redirect_uris here: from_client_secrets_file forwards **kwargs to the
    # flow constructor, which has no such parameter, and passing one raises
    # TypeError before the browser ever opens. run_local_server derives
    # http://localhost:8765 from the port, and Google's desktop-app clients
    # accept any localhost port.
    flow = InstalledAppFlow.from_client_secrets_file(str(secret), scopes=SCOPES)
    url, _ = flow.authorization_url(access_type="offline", prompt="consent")
    print("\n  Opening your browser to grant access.\n  If it does not open, visit:\n")
    print(f"  {url}\n")
    try:
        webbrowser.open(url)
    except Exception:  # pragma: no cover - headless boxes
        pass
    creds = flow.run_local_server(port=8765, prompt_console=False)
    save_credentials(creds)
    return creds


def revoke() -> None:
    s = get_settings()
    if s.token_path.exists():
        s.token_path.unlink()
        log.info("token removed: %s", s.token_path)
    if s.channel_id:
        from ..db import get_db

        get_db().set_meta("channel_id", "")
        log.info("channel_id cleared from db")


def token_age_days(s: Optional[Any] = None) -> Optional[float]:
    """How long ago `auth` last wrote a refresh token, in days.

    The 7-day testing-mode refresh token deadline is not recorded anywhere in
    the token file -- only the access token's own one-hour expiry is -- so the
    only way to warn before a daily cron silently breaks is the file's age.
    """
    import time

    s = s or get_settings()
    if not s.token_path.exists():
        return None
    age = time.time() - s.token_path.stat().st_mtime
    return age / 86400.0


# Google issues refresh tokens that expire after 7 days when the consent screen
# is left in "Testing", and that do not expire once the app is published. The
# failure mode is a cron job that runs for a week and then stops, so it is worth
# warning about well before the deadline rather than at it.
TESTING_REFRESH_TOKEN_DAYS = 7.0
_REAUTH_WARN_DAYS = 6.0


def auth_status() -> Dict[str, Any]:
    s = get_settings()
    secret = credentials_path(s.client_secret_path)
    creds = load_credentials(refresh=False)
    from ..db import get_db

    db = get_db()
    channel_id = s.channel_id or db.get_meta("channel_id")
    age = token_age_days(s)
    return {
        "client_secret": str(secret) if secret else None,
        "token": str(s.token_path) if s.token_path.exists() else None,
        "token_valid": bool(creds and creds.valid),
        "has_refresh_token": bool(creds and creds.refresh_token),
        "scopes": list(creds.scopes) if creds and creds.scopes else [],
        "expiry": creds.expiry.isoformat() if creds and creds.expiry else None,
        "channel_id": channel_id,
        "dry_run": s.dry_run,
        "token_age_days": age,
        # True once the token is old enough that a Testing-mode consent screen
        # may already have invalidated it.
        "reauth_due": age is not None and age >= _REAUTH_WARN_DAYS,
        "testing_token_days": TESTING_REFRESH_TOKEN_DAYS,
    }
