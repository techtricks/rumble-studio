# yt-automation

An unattended pipeline that goes from *news RSS feed* to *scheduled YouTube
upload*: it finds topics, researches them with Claude, writes and validates a
script, narrates it, renders the video with ffmpeg, schedules it, and reports
back what actually performed.

It ships **safe by default**: with no credentials configured it runs fully
offline, plans everything, writes the media to disk, and refuses to call the
YouTube upload API. You turn publishing on deliberately.

```
discover ──▶ script ──▶ approve ──▶ render ──▶ upload ──▶ publish
   ▲            │          │           │          │           │
 topics     validate    sign-off    ffmpeg     schedule    analytics
             LLM        (or auto)    + TTS     (dry run)   feedback
```

---

## Requirements

| | |
|---|---|
| Python | **3.9+** (3.11+ recommended — see [Python version](#python-version-note)) |
| OS | macOS for TTS (`say`); any POSIX for the rest |
| ffmpeg | auto-downloaded by `ytauto setup-ffmpeg`, or set `FFMPEG_PATH` |
| Credentials | optional — only needed to publish or to use the LLM |

The repo vendors nothing large; ffmpeg is fetched on demand into
`vendor/ffmpeg/`.

### Python version note

This project is developed and tested on **Python 3.9.6**, so the code stays
3.9-compatible. Be aware that `google-auth` prints a loud warning on 3.9 because
that Python is past end of life. The CLI filters that specific banner, but the
underlying advice stands: use 3.11+ for anything long-lived.

## Install

```bash
cd yt-automation
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m ytauto setup-ffmpeg     # ~40MB, project-local
.venv/bin/python -m ytauto doctor
```

Then run it as `ytauto <command>` by putting this in your shell profile:

```bash
alias ytauto='.venv/bin/python -m ytauto'
```

## Configuration

```bash
cp .env.example .env     # every value documented inline
```

The defaults are chosen so that a fresh clone is completely inert:

| Switch | Default | Effect |
|---|---|---|
| `DRY_RUN` | `1` | Everything runs except the upload call |
| `AUTO_APPROVE` | `0` | A human must sign off each script |
| `ALLOW_DEGRADED` | `0` | Template scripts (no LLM) are rejected |
| `GEMINI_API_KEY` | unset | No LLM → offline template content |

`DATA_DIR` relocates the database, rendered media and logs together. The
individual `DB_PATH` / `MEDIA_DIR` / `LOG_DIR` variables override it if you
need them scattered.

> **After editing `NICHE`, run `ytauto discover --rescore`.** Scores are
> materialised when a topic is discovered, so topics already in the database
> keep the score they got under the old niche and nothing new will clear the
> threshold. `--rescore` re-applies the current niche to everything unused, and
> needs no network.

## Cost

Worth being blunt, since this is the part people get surprised by:

| | Cost |
|---|---|
| YouTube Data + Analytics API | **Free** — 10,000 units/day. An upload is 1600 |
| Topic sources (RSS, Reddit) | **Free** |
| ffmpeg + `say` rendering | **Free** — local |
| Research and scripting (LLM) | **Free** on the Gemini tier, paid on Anthropic |

The LLM is the only thing that can cost money, and the default config cannot
spend anything: without a key the pipeline runs on templates and refuses to
publish them.

**There is no free Anthropic tier, and a Claude subscription does not include
API access** — subscription and API bill separately. The free path is Google's
Gemini tier, which needs an API key from
[AI Studio](https://aistudio.google.com/apikey) and no billing account:

```bash
echo 'GEMINI_API_KEY=your-key' >> .env
```

`LLM_PROVIDER` picks the backend and defaults to `auto` (Gemini if its key is
present, then Anthropic). Pin it with `LLM_PROVIDER=gemini` or `=anthropic`; a
pinned provider with a missing key fails loudly rather than quietly falling back
to a billable one.

The trade-off is real: Gemini's free models follow a strict JSON schema less
reliably than Claude, so expect a higher rate of scripts rejected by validation
and a lower bar of writing. Raise `LLM_ATTEMPTS` or relax the thresholds in
`.env` before assuming the model is at fault.

## Going live

Three things, in this order.

**1. An LLM key** (optional but strongly recommended — without it you get
template content, not researched content):

```bash
echo 'GEMINI_API_KEY=your-free-key' >> .env   # free tier, no billing
.venv/bin/python -m ytauto llm                # proves the key actually works
```

`ytauto llm` matters: `doctor` can only see that a key *exists*, so a typo
passes it and then fails much later, mid-pipeline. `llm` makes one small real
call and reports the provider, model, latency and what came back — plus whether
the model honoured the real research schema.

### Free providers: set more than one

Every provider below is genuinely free, needs no card and no billing account.
Set as many keys as you like — `auto` takes the one with the most headroom, and
the pipeline hands over to the next when one is rate limited or out of quota.

| Provider | Key | Free allowance | Where |
| --- | --- | --- | --- |
| **Groq** | `GROQ_API_KEY` | tens of thousands of req/day, subject to per-model token limits | [console.groq.com/keys](https://console.groq.com/keys) |
| **NVIDIA NIM** | `NVIDIA_API_KEY` | 40 req/min | [build.nvidia.com](https://build.nvidia.com) |
| Gemini | `GEMINI_API_KEY` | **20 req/day** | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| Anthropic | `ANTHROPIC_API_KEY` | none — paid | — |

Groq is the default pick because Gemini's ceiling is the problem. A live Gemini
quota error reads:

```
quotaId: "GenerateRequestsPerDayPerProjectPerModel-FreeTier", quotaValue: "20"
Please retry in 31.6s.      <-- misleading; the real reset is daily
```

20 `generateContent` calls per day, per model, per project. The "retry in 31s"
is not the real window, and switching models does not help because the project
budget is already spent. At 2–8 calls per video that is a handful of videos a
day — versus hundreds on Groq.

Cerebras was evaluated and rejected: its free tier is a $5, 30-day credit grant
that now requires a verified payment method, with no renewing free allowance.

**Calls per video**, which is why the pipeline is shaped this way:

| Stage | Calls | Why |
| --- | --- | --- |
| research | 1 | one structured call per topic |
| script **+ metadata** | 1 | packaging comes back with the script, because the model just wrote it |
| length beats | 0–6 | only when a long-form script comes back short |

A Short costs 2 calls, a long-form video at most 8. On Groq that is nothing; on
Gemini it is the whole day's budget, which is why Groq leads the `auto` order.
Lower `LONGFORM_MAX_BEATS` to trade video length for more videos per day.

Two failure modes are handled differently, on purpose:

- **429** is per key. Every model on that key is throttled identically, so the
  call waits and is retried; cycling models would spend the budget the retry
  needs. Gemini's misleading "retry in 31s" is exactly this.
- **503** is per model. Capacity is genuinely per-model on these tiers, so that
  one *does* hand over to a different model.

Set `LLM_PROVIDER` explicitly (`groq`, `nvidia`, `gemini`, `anthropic`) to pin
one. You then get that provider or a clear "key not set" error — never a
surprise model and a surprise bill.

**2. YouTube OAuth.** Create a project in the Google Cloud console, enable
*YouTube Data API v3* and *YouTube Analytics API*, create an OAuth client of
type *Desktop app*, then:

```bash
cp ~/Downloads/client_secret_*.json secrets/client_secret.json
.venv/bin/python -m ytauto auth        # opens a browser, one-time
```

`auth` writes `secrets/token.json` and discovers your channel id. The token is
refreshed automatically afterwards.

**3. Flip the switches:**

```bash
DRY_RUN=0
AUTO_APPROVE=1
```

```bash
.venv/bin/python -m ytauto doctor   # must exit 0 now
.venv/bin/python -m ytauto run --cycles 1
```

`doctor` exits non-zero and lists exactly what is missing, so it is safe to wire
into a boot script or cron:

```bash
.venv/bin/python -m ytauto doctor || .venv/bin/python -m ytauto run
```

### Being fully unattended

`AUTO_APPROVE=1` is the switch that removes the human from the loop. It only
auto-approves scripts that passed every validation gate *and* were written by a
real LLM. A degraded template script is never auto-approved, whatever else is
configured — `ALLOW_DEGRADED` cannot override that, by design.

To run continuously:

```cron
*/30 * * * * cd /path/to/yt-automation && .venv/bin/python -m ytauto run >> data/logs/cron.log 2>&1
```

## Commands

| Command | What it does |
|---|---|
| `status` | Channel, quota, pipeline queue and calendar in one screen |
| `doctor` | Everything needed to publish; non-zero exit if not ready |
| `llm` | Make one real API call to prove the key and model work |
| `auth` | Grant or revoke YouTube access |
| `discover` | Harvest and score topics (`--show`/`--rescore` are offline, `--add` is manual) |
| `script` | Research + write + validate (`--review` lists what's waiting) |
| `approve` | Sign off scripted videos (`--force` overrides validation) |
| `render` | Narrate and composite approved videos |
| `upload` | Upload and schedule (`--sync` verifies what went live) |
| `publish` | Confirm which scheduled videos are actually public |
| `analytics` | Performance report (`--fetch` costs 1000 quota units) |
| `run` | The full unattended loop; `--cycles N` to repeat |
| `quota` | Today's YouTube API spend against the daily cap |
| `setup-ffmpeg` | Download a project-local static ffmpeg |
| `logs` | Tail the log file |

`--help` on any of them lists the flags. Every command is safe to interrupt and
re-run: work is claimed with leases and each stage is idempotent, so a crash
mid-render does not leave a half-published video.

## Monetisation safety

A channel that cannot be monetised is a hobby, so this is treated as a hard
requirement rather than a quality goal. YouTube's Partner Program pays for
"original and authentic" content, and the 15 July 2025 clarification renamed the
old *repetitious content* rule to **inauthentic content** and made the examples
concrete. Four of them describe what an unattended tech-news pipeline would do
by default, so the pipeline is built against them explicitly:

| YouTube's example | What this project does |
| --- | --- |
| "Reading materials you didn't create (e.g., articles from websites)" | The script prompt requires the channel's own analysis per beat, and `check_originality` measures the script against the source article: too much shared wording, or too little vocabulary the article never used, is refused. |
| "Image slideshows, templated storylines … with minimal … educational value" | Slideshow slides are sourced pictures, but the narration carries the argument; images never replace commentary. |
| "channels where all slideshows use the same voiceover across videos" | The narration voice rotates per video (`TTS_VOICE_POOL`), deterministically by video id so a re-render is reproducible. |
| "AI-generated content made with generic or unoriginal templates" | Scripts are checked against everything already published; near-duplicates are blocked. |

Enforcement is **channel level**: one flagged video can cost a channel its
monetisation, not just that upload. So these checks refuse rather than warn. A
video that fails stays unuploaded and the reason is written to
`videos.policy_error`.

**Copyright.** Copyrighted news photographs are off by default
(`ALLOW_ARTICLE_IMAGES=0`). Only CC0, PDM, CC BY and CC BY-SA pictures are used,
because the channel is commercial and the renderer crops and scales every
frame — NC forbids monetisation and ND forbids the crop. Openverse is queried
with `license_type=commercial,modification` for the same reason. An unrecognised
licence is treated as a refusal, not a guess.

**How "rewriting an article" is detected.** Not by counting the word
"however" — that measures the prompt rather than the script, and it fails a
script that reasons well in different words. `source_overlap` compares the
script to the article it came from: a restatement can only re-use the source's
own phrasing, so it scores high on shared 4-grams and low on vocabulary the
source never used. Connecting two facts requires words the article did not
contain, so genuine analysis is low on the first and high on the second. Both
numbers must be out of bounds before the script is refused.

**Attribution.** CC BY and CC BY-SA require the credit to travel with the work,
so credits go into the video's YouTube description, not only into
`data/media/<id>/credits.txt`. `check_assets` refuses an image with a licence
but no attribution.

**Synthetic content must be disclosed.** The narration is a computer voice and
some visuals are sourced photographs, so `build_description` states this in the
description of every upload. The matching **altered-content** toggle lives in
YouTube Studio and has no API surface, so it is a manual step — see
`ytauto.policy.manual_steps()`.

Also required before money arrives, none of which this code can do for you: 500
subscribers and 4,000 watch hours, an AdSense account in a supported country,
and — because a template-driven channel is the exact profile these rules target
— a human who actually reads the script before approving it. The pipeline leaves
`AUTO_APPROVE=0` on for that reason.

## How the safety actually works

- **Quota.** Upload costs 1600 units against a 10,000/day cap. The ledger is
  written before the call, not after, so a crash cannot under-count spend.
  Planning never exceeds 90% of the cap.
- **Validation.** Every script must clear niche relevance, fact count, research
  confidence, title quality, and A/B title distinctness. Failures are reported
  per-rule with the specific reason.
- **Dry run.** Uploads are simulated end to end: the real request is built and
  logged, the schedule slot is chosen from real calendar state, and the database
  is left untouched.
- **Isolation.** Tests can never reach your data or the network — the whole
  suite runs against a temporary `DATA_DIR` with `urllib` and `requests`
  blocked at the socket layer.

## Layout

```
ytauto/
  config.py            settings, .env, provider choice, paths, quota costs
  llm.py               provider dispatch, JSON schemas, offline fallback
  llm_gemini.py        Gemini backend over REST (free, 20 req/day)
  llm_openai_compat.py Groq + NVIDIA NIM backends (free tier)
  db.py                schema, state machine, scoring, leases, quota ledger
  cli.py               every command
  content.py           Claude research, scripts, metadata, offline fallback
  validate.py          the publish gates
  schedule.py          timezone-aware slot planning
  sources/             RSS, Reddit
  render/              TTS, captions, ffmpeg composition, thumbnails
  youtube/             Data + Analytics clients, OAuth
  pipeline/            the individual stages and the unattended loop
tests/                 188 fast tests + 10 real-ffmpeg tests
vendor/ffmpeg/         fetched on demand
data/                  database, media, logs (gitignored)
secrets/               OAuth client + token (gitignored)
```

## Tests

```bash
.venv/bin/python -m pytest tests/ -q -m "not slow"   # 271 fast, ~2s
.venv/bin/python -m pytest tests/ -q                 # 281 total, ~60s
```

The fast suite is hermetic — temporary database per test, no network, no real
ffmpeg. The `slow` tests shell out to the real ffmpeg binary and macOS `say`,
and are the ones that catch encoder, font and filter-expression problems that a
mocked pipeline cannot.

## Troubleshooting

**`doctor` exits 1** — run it; it prints the specific blocker (missing OAuth
client, missing token, missing API key) rather than a generic failure.

**Everything says "written without an LLM"** — no LLM key is visible to the
process actually running. `ytauto status` shows the provider and model it
resolved, which is the fastest way to see what it picked.

**`quota` says the daily cap is reached** — uploads cost 1600 units. Wait for
the UTC day to roll over, or raise `YOUTUBE_DAILY_QUOTA` if you have more.

**A video is `failed` and won't re-render** — `ytauto script --review` shows
the rule that failed. Fix and re-run with `--force`, or `ytauto render --force`.

**Tests touch my real database** — they shouldn't; this was a bug, fixed in the
per-test isolation fixture. If you ever see it again, check that no `DATA_DIR`
is exported in your shell, since an inherited value wins over the test's.
