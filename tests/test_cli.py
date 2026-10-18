"""CLI surface: argument wiring, exit codes, and output contract.

`main()` is called directly rather than via a subprocess so the suite stays
fast; a couple of subprocess checks confirm the entry point really works.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ytauto.cli import main

ROOT = Path(__file__).resolve().parent.parent


def run(*argv) -> int:
    return main(list(argv))


# ------------------------------------------------------------------- basics
def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        run("--help")
    assert exc.value.code == 0
    assert "research" in capsys.readouterr().out


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        run("--version")
    assert exc.value.code == 0
    assert "ytauto" in capsys.readouterr().out


def test_no_command_is_an_error():
    with pytest.raises(SystemExit) as exc:
        run()
    assert exc.value.code != 0


# ------------------------------------------------------------------ doctor
def test_doctor_reports_missing_credentials(capsys):
    """Exit non-zero when it cannot publish: this is the command a user runs
    before trusting the tool, so a false "ready" is the worst possible bug."""
    code = run("doctor")
    out = capsys.readouterr().out
    assert code == 1
    assert "OAuth client" in out
    assert "ANTHROPIC_API_KEY" in out


def test_doctor_confirms_ffmpeg_and_tts(capsys):
    run("doctor")
    out = capsys.readouterr().out
    assert "ffmpeg" in out
    assert "tts" in out


def test_doctor_surfaces_the_safety_switches(capsys):
    run("doctor")
    out = capsys.readouterr().out
    # dry_run and auto_approve must both be visible, whatever they are set to:
    # they are the two switches that decide whether words reach the channel.
    assert "dry_run" in out
    assert "auto_approve" in out


# ------------------------------------------------------------------ status
def test_status_runs_offline(capsys):
    assert run("status") == 0
    out = capsys.readouterr().out
    assert "channel" in out
    assert "quota" in out
    assert "upcoming" in out


def test_status_shows_channel_id_as_unset(capsys):
    run("status")
    assert "not detected" in capsys.readouterr().out


# -------------------------------------------------------------------- auth
def test_auth_status_is_json(capsys):
    assert run("auth", "--status") == 0
    data = json.loads(capsys.readouterr().out)
    assert "token_valid" in data
    assert data["token_valid"] is False


def test_auth_without_a_client_fails_cleanly(capsys):
    code = run("auth")
    assert code == 1
    assert "client_secret" in capsys.readouterr().out


# ----------------------------------------------------------------- discover
def test_discover_show_is_offline(capsys):
    assert run("discover", "--show", "--limit", "5") == 0
    assert "top topics" in capsys.readouterr().out


def test_discover_add_manual(capsys):
    assert run("discover", "--add", "A new AI story", "--summary", "About a gpu.") == 0
    assert "added" in capsys.readouterr().out


def test_discover_add_manual_rejects_a_blank_title(capsys):
    assert run("discover", "--add", "   ") == 1
    assert "non-empty" in capsys.readouterr().out


# ------------------------------------------------------- script / approve
def test_script_review_with_nothing_pending(capsys):
    assert run("script", "--review") == 0
    assert "awaiting review" in capsys.readouterr().out


def test_script_without_topics_is_a_soft_failure(capsys):
    code = run("script", "--count", "1")
    out = capsys.readouterr().out
    assert code == 1
    assert "discover" in out


def test_approve_with_nothing_pending(capsys):
    assert run("approve") == 0
    assert "nothing awaiting" in capsys.readouterr().out


# ------------------------------------------------------------------ render
def test_render_with_an_empty_queue(capsys):
    assert run("render") == 0
    assert "no approved videos" in capsys.readouterr().out


def test_render_unknown_id_is_reported(capsys):
    assert run("render", "--id", "deadbeef") == 1
    assert "unknown video" in capsys.readouterr().out


# ------------------------------------------------------------------ upload
def test_upload_with_an_empty_queue(capsys):
    assert run("upload") == 0
    assert "no rendered videos" in capsys.readouterr().out


def test_upload_unknown_id(capsys):
    assert run("upload", "--id", "deadbeef") == 1
    assert "unknown video" in capsys.readouterr().out


def test_dry_run_flag_is_wired_both_ways(capsys):
    """--dry-run / --no-dry-run must be mutually exclusive, not silently
    last-one-wins."""
    with pytest.raises(SystemExit):
        run("upload", "--dry-run", "--no-dry-run")


def test_quota_command_runs_offline(capsys):
    assert run("quota") == 0
    out = capsys.readouterr().out
    assert "remaining" in out
    assert "no API calls recorded yet" in out


# ---------------------------------------------------------------- analytics
def test_analytics_report_runs_offline(capsys):
    assert run("analytics") == 0
    out = capsys.readouterr().out
    assert "views" in out
    assert "top videos" in out


def test_analytics_fetch_is_refused_without_uploads(capsys):
    assert run("analytics", "--fetch") == 0
    assert "0 analytics row" in capsys.readouterr().out


# ----------------------------------------------------------------- others
def test_logs_prints_the_tail(tmp_path, monkeypatch, capsys):
    """`cmd_logs` resolves the directory at call time, so the file can be
    staged directly. (The rotating handler itself binds at setup, which is
    correct for a real process and only awkward inside a shared test one.)"""
    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    logfile = tmp_path / "ytauto.log"
    logfile.write_text("\n".join(f"line {i}" for i in range(200)), encoding="utf-8")
    assert run("logs", "--lines", "5") == 0
    out = capsys.readouterr().out
    assert "line 199" in out
    assert "line 195" in out
    assert "line 194" not in out        # honours --lines


def test_logs_handles_a_missing_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "does-not-exist"))
    assert run("logs") == 1
    assert "no log file" in capsys.readouterr().out


def test_setup_ffmpeg_reports_the_existing_binary(capsys):
    assert run("setup-ffmpeg") == 0
    assert "already at" in capsys.readouterr().out


# ------------------------------------------------------------ entry point
def test_module_entry_point_runs():
    """`python -m ytauto` is the documented interface; make sure it works."""
    proc = subprocess.run(
        [sys.executable, "-m", "ytauto", "status"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "channel" in proc.stdout


def test_entry_point_hides_the_eol_warning():
    """google-auth's FutureWarning used to print in front of every command."""
    proc = subprocess.run(
        [sys.executable, "-m", "ytauto", "quota"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=60,
    )
    assert "past its end of life" not in proc.stderr


def test_discover_rescore_reapplies_the_niche(capsys, monkeypatch):
    """`discover --rescore` must not need the network and must report a count."""
    from ytauto.db import get_db

    db = get_db()
    db.add_topic("Nvidia ships a faster GPU", "manual", url="https://e.com/a",
                 summary="Nvidia announced a new datacenter GPU. " * 3)
    db.score_topics(["unrelated", "words"], half_life_h=24.0)
    assert run("discover", "--rescore") == 0
    assert "rescored 1 topic" in capsys.readouterr().out

    monkeypatch.setenv("NICHE", "gpu,nvidia")
    assert run("discover", "--rescore") == 0
    assert db.get_topic(db.top_topics(limit=1)[0]["id"])["niche_score"] > 0.0


def test_discover_show_limit_defaults_sensibly(capsys):
    assert run("discover", "--show") == 0
    assert "top topics" in capsys.readouterr().out


# ------------------------------------------------------------------ llm check
def test_llm_reports_the_missing_key(capsys, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "auto")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert run("llm") == 1
    assert "GEMINI_API_KEY" in capsys.readouterr().out


def test_llm_check_round_trip(capsys, monkeypatch):
    """A good response must be reported as working, with the payload shown."""
    seen = {}

    def _complete(self, prompt, schema, *, system="", model=None, max_tokens=None):
        seen["schema"] = schema
        seen["prompt"] = prompt
        return {
            "angle": "Blackwell doubles inference throughput",
            "key_facts": ["ships in 2026", "2x throughput"],
            "confidence": 0.8,
        }

    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr("ytauto.llm.LLM.complete", _complete)

    assert run("llm") == 0
    out = capsys.readouterr().out
    assert "working" in out
    assert "Blackwell" in out
    assert seen["schema"]["required"] == ["angle", "key_facts", "confidence"]


def test_llm_check_fails_when_the_schema_is_ignored(capsys, monkeypatch):
    """A model that ignores the schema must be reported as broken, not OK."""
    from ytauto import cli as cli_mod

    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr(
        "ytauto.llm.LLM.complete",
        lambda self, p, s, **kw: {"angle": "x", "key_facts": "not a list",
                                  "confidence": 0.5},
    )
    assert run("llm") == 1
    assert "schema" in capsys.readouterr().out.lower()


def test_llm_check_surfaces_a_call_error(capsys, monkeypatch):
    from ytauto.llm import LLMError

    def _boom(self, prompt, schema, *, system="", model=None, max_tokens=None):
        raise LLMError("HTTP 400: API key not valid")

    monkeypatch.setenv("GEMINI_API_KEY", "bad-key")
    monkeypatch.setattr("ytauto.llm.LLM.complete", _boom)
    assert run("llm") == 1
    assert "API key not valid" in capsys.readouterr().out
