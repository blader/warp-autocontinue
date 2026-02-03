#!/usr/bin/env python3
"""warp-autocontinue

A best-effort local hook for Warp Desktop Agent sessions.

It monitors Warp's on-disk sqlite DB and, when it detects a new assistant message
in the most-recent agent conversation, it evaluates whether the agent is done.
If the agent looks paused/incomplete (progress update / question / error), it
automatically sends the user message: "Please continue".

Notes:
- This relies on private/internal Warp storage and may break across Warp updates.
- Auto-send uses AppleScript (System Events) and requires Accessibility perms.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# -----------------
# Constants / config
# -----------------

# Warp stores on-disk data in a macOS Group Container.
# This is a stable identifier for Warp.
WARP_GROUP_CONTAINER_ID = "2BBY89MBSN.dev.warp"

# Warp DB location is fixed per channel.
WARP_APP_SUPPORT_SUBPATH = Path(
    f"Library/Group Containers/{WARP_GROUP_CONTAINER_ID}/Library/Application Support"
)
WARP_DB_PREVIEW_SUBPATH = Path("dev.warp.Warp-Preview/warp.sqlite")
WARP_DB_STABLE_SUBPATH = Path("dev.warp.Warp-Stable/warp.sqlite")

DEFAULT_WARP_BUNDLE_ID = "dev.warp.Warp-Preview"
DEFAULT_STATE_SUBPATH = Path("Library/Application Support/warp-autocontinue/state.json")
DEFAULT_CONTINUE_MESSAGE = "Please continue"

# macOS virtual keycode for the Return/Enter key.
MACOS_KEYCODE_RETURN = 36

# Heuristic evaluator keywords.
HEURISTIC_ERROR_MARKERS = (
    "traceback",
    "error:",
    "failed",
    "exception",
    "exit_code",
)
HEURISTIC_WAITING_MARKERS = (
    "before i",
    "i need",
    "need your",
    "can you",
    "could you",
    "please confirm",
    "please tell me",
    "which one",
    "do you want",
    "should i",
    "let me know",
    "what should",
    "clarif",
    "i can proceed",
    "waiting for",
)
HEURISTIC_PROGRESS_MARKERS = (
    "next i will",
    "next, i",
    "next step",
    "i will now",
    "i'm going to",
    "i am going to",
    "still need to",
    "remaining",
    "not finished",
    "in progress",
    "working on",
    "investigating",
    "todo",
)
HEURISTIC_DONE_MARKERS = (
    "all set",
    "done",
    "completed",
    "finished",
    "wrapped up",
)

OPENAI_EVAL_SYSTEM_PROMPT = (
    "You are a strict evaluator for an agentic coding session. "
    "Given recent user messages, an optional plan, and the last assistant response, "
    "decide if the assistant response is a final, complete deliverable. "
    "If it is incomplete/progress-only/asking questions/blocked by errors, choose continue. "
    "If it is complete and no further work is needed, choose stop. "
    'Respond ONLY as JSON: {"action": "continue"|"stop", "reason": "..."}.'
)


# Message extraction from `agent_tasks.task` relies on a stable-but-undocumented
# protobuf-ish framing. These constants make the reverse-engineering assumptions
# explicit and easy to update if Warp changes its internal representation.
UUID_ASCII_LEN = 36
TASK_BLOB_TAG_USER_TEXT = 0x12
TASK_BLOB_TAG_ASSISTANT_TEXT = 0x1A
TASK_BLOB_TAG_STRING = 0x0A

# Regex used to find UUID framing in Warp's agent task blobs.
UUID_RE_BYTES = re.compile(
    rb"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)


@dataclass(frozen=True)
class AgentTask:
    conversation_id: str
    task_id: str
    last_modified_at: str
    task_blob: bytes


@dataclass(frozen=True)
class ExtractedMessage:
    offset: int
    uuid: str
    role: str  # "user" | "assistant"
    text: str


@dataclass(frozen=True)
class PlanDoc:
    notebook_id: int
    ai_document_id: str
    title: str
    markdown: str


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def locate_warp_db_path(explicit: Optional[str] = None) -> Path:
    if explicit:
        p = Path(explicit).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"warp sqlite db not found at: {p}")
        return p

    app_support = Path.home() / WARP_APP_SUPPORT_SUBPATH

    # Warp stores its sqlite DB in a fixed location per channel.
    # We intentionally keep this simple and only check Preview then Stable.
    preview = app_support / WARP_DB_PREVIEW_SUBPATH
    if preview.exists():
        return preview

    stable = app_support / WARP_DB_STABLE_SUBPATH
    if stable.exists():
        return stable

    raise FileNotFoundError(
        "Could not find Warp sqlite DB. Tried:\n"
        f"- {preview}\n"
        f"- {stable}\n\n"
        "If you use a different Warp channel/build, pass --db-path (or set WARP_DB_PATH)."
    )


def open_warp_db_ro(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 2000")
    return conn


def get_latest_agent_task(
    conn: sqlite3.Connection, conversation_id: Optional[str] = None
) -> Optional[AgentTask]:
    if conversation_id:
        row = conn.execute(
            "SELECT conversation_id, task_id, task, last_modified_at "
            "FROM agent_tasks WHERE conversation_id = ? "
            "ORDER BY last_modified_at DESC LIMIT 1",
            (conversation_id,),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT conversation_id, task_id, task, last_modified_at "
            "FROM agent_tasks ORDER BY last_modified_at DESC LIMIT 1"
        ).fetchone()

    if not row:
        return None

    return AgentTask(
        conversation_id=row["conversation_id"],
        task_id=row["task_id"],
        last_modified_at=str(row["last_modified_at"]),
        task_blob=row["task"],
    )


def read_varint(buf: bytes, i: int) -> tuple[int, int]:
    """Decode a protobuf varint starting at offset i.

    Returns: (value, next_offset)

    We only need a tiny subset of protobuf parsing to extract strings from
    Warp's `agent_tasks.task` blob.
    """

    shift = 0
    val = 0
    while i < len(buf):
        b = buf[i]
        val |= (b & 0x7F) << shift
        i += 1
        if not (b & 0x80):
            return val, i
        shift += 7
        if shift > 70:
            raise ValueError("varint too long")
    raise ValueError("unexpected EOF reading varint")


def _parse_uuid_framed_message_at(buf: bytes, pos: int) -> Optional[ExtractedMessage]:
    """Try to parse a single user/assistant message anchored at a UUID.

    Observed structure inside Warp's `agent_tasks.task` (BLOB):
    - At `pos`: a UUID encoded as 36 ASCII bytes.
    - Immediately after: a protobuf-ish field tag distinguishing user vs assistant.
    - Next: a length-delimited nested message whose first field is a UTF-8 string.

    If Warp changes this framing, message extraction will fail *silently* (None)
    and the tool will effectively no-op.
    """

    uuid_bytes = buf[pos : pos + UUID_ASCII_LEN]
    if len(uuid_bytes) != UUID_ASCII_LEN:
        return None

    tag_pos = pos + UUID_ASCII_LEN
    if tag_pos >= len(buf):
        return None
    tag = buf[tag_pos]
    if tag not in (TASK_BLOB_TAG_USER_TEXT, TASK_BLOB_TAG_ASSISTANT_TEXT):
        return None

    try:
        chunk_len, j = read_varint(buf, tag_pos + 1)
    except Exception:
        return None

    if chunk_len <= 0 or j + chunk_len > len(buf):
        return None
    chunk = buf[j : j + chunk_len]

    if not chunk or chunk[0] != TASK_BLOB_TAG_STRING:
        return None

    try:
        str_len, k = read_varint(chunk, 1)
    except Exception:
        return None

    if str_len <= 0 or k + str_len > len(chunk):
        return None

    sbytes = chunk[k : k + str_len]
    try:
        text = sbytes.decode("utf-8")
    except UnicodeDecodeError:
        return None

    text = text.strip()
    if len(text) < 2:
        return None

    role = "user" if tag == TASK_BLOB_TAG_USER_TEXT else "assistant"
    return ExtractedMessage(
        offset=pos,
        uuid=uuid_bytes.decode("ascii", "ignore"),
        role=role,
        text=text,
    )


def extract_messages_from_agent_task_blob(blob: bytes) -> list[ExtractedMessage]:
    """Extract user/assistant messages from a Warp `agent_tasks.task` blob.

    This is intentionally best-effort and relies on internal storage details.
    We scan for UUIDs first to avoid attempting to fully decode the protobuf.
    """

    msgs: list[ExtractedMessage] = []
    for m in UUID_RE_BYTES.finditer(blob):
        em = _parse_uuid_framed_message_at(blob, m.start())
        if em:
            msgs.append(em)
    msgs.sort(key=lambda x: x.offset)
    return msgs


def get_last_assistant_message(
    msgs: list[ExtractedMessage],
) -> Optional[ExtractedMessage]:
    for m in reversed(msgs):
        if m.role == "assistant":
            return m
    return None


def get_recent_user_messages(
    msgs: list[ExtractedMessage], max_messages: int
) -> list[str]:
    out: list[str] = []
    for m in reversed(msgs):
        if m.role != "user":
            continue
        out.append(m.text)
        if len(out) >= max_messages:
            break
    out.reverse()
    return out


def get_latest_plan_doc_for_blob(
    conn: sqlite3.Connection, blob: bytes
) -> Optional[PlanDoc]:
    """Best-effort extraction of the in-context Plan for a given agent task blob.

    Warp's plan documents appear in sqlite as:
    - `ai_document_panes.document_id` (the logical plan document id)
    - `notebooks.ai_document_id` (key to the markdown stored in `notebooks.data`)

    The agent task blob embeds the document id; we use that as our join key.
    """

    # Plans created by the agent are stored as notebooks where notebooks.ai_document_id
    # matches ai_document_panes.document_id.
    panes = conn.execute(
        "SELECT id, document_id, version FROM ai_document_panes ORDER BY id DESC"
    ).fetchall()

    for pane in panes:
        doc_id = pane["document_id"]
        if not doc_id:
            continue
        if doc_id.encode("utf-8") not in blob:
            continue

        nb = conn.execute(
            "SELECT id, title, data, ai_document_id FROM notebooks WHERE ai_document_id = ?",
            (doc_id,),
        ).fetchone()
        if not nb:
            continue

        title = nb["title"] or "(untitled plan)"
        data = nb["data"] or ""
        return PlanDoc(
            notebook_id=int(nb["id"]),
            ai_document_id=str(nb["ai_document_id"]),
            title=str(title),
            markdown=str(data),
        )

    return None


def load_state(state_path: Path) -> dict[str, Any]:
    """Load the debounce state file.

    Schema (best-effort, may evolve):
    {
      "conversations": {
        "<conversation_id>": {
          "last_assistant_uuid": "...",
          "last_action": "continue"|"stop",
          "last_reason": "...",
          "last_modified_at": "...",
          "sent_continue": true|false,
          "sent_continue_ts": <float epoch seconds>,
          "sent_continue_result": "dry_run"|"ok"|"skipped"
        }
      }
    }

    If the file is missing/corrupt, we treat it as empty.
    """

    if not state_path.exists():
        return {"conversations": {}}
    try:
        return json.loads(state_path.read_text())
    except Exception:
        return {"conversations": {}}


def save_state(state_path: Path, state: dict[str, Any]) -> None:
    """Atomically write state.json.

    We write to a temp file and rename to avoid partially-written JSON.
    """

    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(state_path)


def frontmost_bundle_id() -> str:
    # Works on macOS via System Events.
    out = subprocess.check_output(
        [
            "osascript",
            "-e",
            'tell application "System Events" to get bundle identifier of first application process whose frontmost is true',
        ],
        text=True,
    )
    return out.strip()


def send_user_message_via_osascript(
    *,
    warp_bundle_id: str,
    message: str,
    require_frontmost: bool,
    allow_activate: bool,
) -> bool:
    """Type a message into Warp's input editor and press Return.

    This uses AppleScript/System Events, which has no semantic understanding of
    "which text box" it's typing into.

    Safety model:
    - If require_frontmost=True (default), we only type if Warp is already
      frontmost.
    - If allow_activate=True, we first activate Warp and then verify it's
      frontmost before typing.
    - Otherwise, we do nothing (too risky).
    """
    if require_frontmost:
        if frontmost_bundle_id() != warp_bundle_id:
            return False
    elif allow_activate:
        subprocess.run(
            [
                "osascript",
                "-e",
                f'tell application id "{warp_bundle_id}" to activate',
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        time.sleep(0.15)
        if frontmost_bundle_id() != warp_bundle_id:
            return False
    else:
        # Unsafe: would type into whatever app is frontmost.
        return False

    # Type and press Return.
    script = (
        'tell application "System Events"\n'
        f"  keystroke {json.dumps(message)}\n"
        f"  key code {MACOS_KEYCODE_RETURN}\n"
        "end tell\n"
    )
    res = subprocess.run(
        ["osascript", "-e", script],
        text=True,
        capture_output=True,
    )
    return res.returncode == 0


def heuristic_should_continue(*, assistant_text: str) -> tuple[bool, str]:
    t = assistant_text.strip()
    tl = t.lower()

    # Errors / failures: tend to need another attempt.
    if any(x in tl for x in HEURISTIC_ERROR_MARKERS):
        return True, "assistant message looks like an error"

    # If it's asking a question, treat as paused.
    if "?" in t:
        return True, "assistant asked a question"

    # Common “waiting for user” phrases.
    if any(x in tl for x in HEURISTIC_WAITING_MARKERS):
        return True, "assistant appears to be waiting for user input"

    # Progress / unfinished work signals.
    if any(x in tl for x in HEURISTIC_PROGRESS_MARKERS):
        return True, "assistant appears to be mid-task/progress-only"

    # If it looks like a final summary, stop.
    if any(x in tl for x in HEURISTIC_DONE_MARKERS):
        return False, "assistant appears to be done"

    # Default: do nothing.
    return False, "no strong signal to continue"


def openai_eval(
    *,
    api_key: str,
    model: str,
    prompt: str,
    assistant_text_for_heuristic: str,
    base_url: str = "https://api.openai.com/v1",
    timeout_s: int = 30,
) -> tuple[str, str]:
    """Return (action, reason) where action is 'continue' or 'stop'."""

    url = base_url.rstrip("/") + "/chat/completions"

    system = OPENAI_EVAL_SYSTEM_PROMPT

    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8")
    data = json.loads(raw)
    content = data["choices"][0]["message"]["content"]

    try:
        out = json.loads(content)
    except Exception:
        # If the model returned non-JSON, fall back to heuristic on the assistant response.
        ok, h_reason = heuristic_should_continue(
            assistant_text=assistant_text_for_heuristic
        )
        return ("continue" if ok else "stop"), f"openai returned non-JSON; {h_reason}"

    action = str(out.get("action", "")).strip().lower()
    reason = str(out.get("reason", "")).strip()
    if action not in ("continue", "stop"):
        ok, h_reason = heuristic_should_continue(
            assistant_text=assistant_text_for_heuristic
        )
        return ("continue" if ok else "stop"), f"openai invalid action; {h_reason}"

    return action, (reason or "(no reason)")


def build_eval_prompt(
    *,
    user_messages: list[str],
    assistant_message: str,
    plan: Optional[PlanDoc],
    max_plan_chars: int,
) -> str:
    parts: list[str] = []

    parts.append("Recent user messages (oldest -> newest):")
    if user_messages:
        for i, um in enumerate(user_messages, 1):
            parts.append(f"{i}. {um}")
    else:
        parts.append("(none found)")

    parts.append("")
    if plan:
        plan_text = plan.markdown
        if len(plan_text) > max_plan_chars:
            plan_text = plan_text[:max_plan_chars] + "\n…(truncated)…"
        parts.append(f"In-context Plan: {plan.title}")
        parts.append(plan_text)
    else:
        parts.append("In-context Plan: (none)")

    parts.append("")
    parts.append("Last assistant response:")
    parts.append(assistant_message)

    return "\n".join(parts)


def evaluate_action(
    *,
    evaluator: str,
    prompt: str,
    assistant_text: str,
) -> tuple[str, str]:
    """Return (action, reason).

    action is the string "continue" or "stop".

    This is deliberately strict: anything that looks like a partial answer,
    a progress update, an error, or a question should become "continue".
    """

    if evaluator == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        model = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini").strip()
        if not api_key:
            ok, h_reason = heuristic_should_continue(assistant_text=assistant_text)
            return "continue" if ok else "stop", f"OPENAI_API_KEY not set; {h_reason}"

        try:
            return openai_eval(
                api_key=api_key,
                model=model,
                prompt=prompt,
                assistant_text_for_heuristic=assistant_text,
            )
        except (urllib.error.URLError, KeyError, ValueError) as e:
            ok, h_reason = heuristic_should_continue(assistant_text=assistant_text)
            action = "continue" if ok else "stop"
            return action, f"openai error: {e}; {h_reason}"

    ok, reason = heuristic_should_continue(assistant_text=assistant_text)
    return ("continue" if ok else "stop"), reason


def maybe_send_continue(
    *,
    warp_bundle_id: str,
    allow_activate: bool,
    require_frontmost: bool,
    dry_run: bool,
) -> str:
    """Send the default continue message.

    Returns a small string suitable for writing to state.json.
    """

    if dry_run:
        return "dry_run"

    ok = send_user_message_via_osascript(
        warp_bundle_id=warp_bundle_id,
        message=DEFAULT_CONTINUE_MESSAGE,
        require_frontmost=require_frontmost,
        allow_activate=allow_activate,
    )
    return "ok" if ok else "skipped"


def decide_and_maybe_continue(
    *,
    db_path: Path,
    warp_bundle_id: str,
    conversation_id: Optional[str],
    state_path: Path,
    max_user_messages: int,
    max_plan_chars: int,
    evaluator: str,
    allow_activate: bool,
    require_frontmost: bool,
    dry_run: bool,
) -> None:
    conn = open_warp_db_ro(db_path)
    try:
        task = get_latest_agent_task(conn, conversation_id)
        if not task:
            return

        msgs = extract_messages_from_agent_task_blob(task.task_blob)
        last_assistant = get_last_assistant_message(msgs)
        if not last_assistant:
            return

        state = load_state(state_path)
        conv_state = state.setdefault("conversations", {}).setdefault(
            task.conversation_id, {}
        )
        last_seen_uuid = conv_state.get("last_assistant_uuid")
        if last_seen_uuid == last_assistant.uuid:
            return

        plan = get_latest_plan_doc_for_blob(conn, task.task_blob)
        recent_users = get_recent_user_messages(msgs, max_user_messages)

        prompt = build_eval_prompt(
            user_messages=recent_users,
            assistant_message=last_assistant.text,
            plan=plan,
            max_plan_chars=max_plan_chars,
        )

        action, reason = evaluate_action(
            evaluator=evaluator,
            prompt=prompt,
            assistant_text=last_assistant.text,
        )

        conv_state["last_assistant_uuid"] = last_assistant.uuid
        conv_state["last_action"] = action
        conv_state["last_reason"] = reason
        conv_state["last_modified_at"] = task.last_modified_at

        if action == "continue":
            conv_state["sent_continue"] = True
            conv_state["sent_continue_ts"] = time.time()
            conv_state["sent_continue_result"] = maybe_send_continue(
                warp_bundle_id=warp_bundle_id,
                allow_activate=allow_activate,
                require_frontmost=require_frontmost,
                dry_run=dry_run,
            )
        else:
            conv_state["sent_continue"] = False

        save_state(state_path, state)
    finally:
        conn.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="warp-autocontinue",
        description="Auto-send 'Please continue' when Warp's agent pauses with an incomplete response.",
    )

    parser.add_argument(
        "--db-path",
        default=os.environ.get("WARP_DB_PATH"),
        help="Path to warp.sqlite (optional; auto-detected by default).",
    )
    parser.add_argument(
        "--warp-bundle-id",
        default=os.environ.get("WARP_BUNDLE_ID", DEFAULT_WARP_BUNDLE_ID),
        help=(
            "Warp bundle identifier used for frontmost checks "
            f"(default: {DEFAULT_WARP_BUNDLE_ID})."
        ),
    )
    parser.add_argument(
        "--conversation-id",
        default=os.environ.get("WARP_CONVERSATION_ID"),
        help="If set, only monitor this agent conversation_id.",
    )
    parser.add_argument(
        "--state-path",
        default=os.environ.get(
            "WARP_AUTOCONTINUE_STATE",
            str(Path.home() / DEFAULT_STATE_SUBPATH),
        ),
        help=(
            "State file path for debouncing "
            "(default: ~/Library/Application Support/warp-autocontinue/state.json)."
        ),
    )
    parser.add_argument(
        "--max-user-messages",
        type=int,
        default=int(os.environ.get("WARP_AUTOCONTINUE_MAX_USER", "4")),
        help="How many recent user messages to include in evaluation.",
    )
    parser.add_argument(
        "--max-plan-chars",
        type=int,
        default=int(os.environ.get("WARP_AUTOCONTINUE_MAX_PLAN_CHARS", "8000")),
        help="Max plan chars to include (truncate beyond this).",
    )
    parser.add_argument(
        "--evaluator",
        choices=["heuristic", "openai"],
        default=os.environ.get("WARP_AUTOCONTINUE_EVAL", "heuristic"),
        help="Evaluator to decide whether to continue.",
    )

    frontmost_group = parser.add_mutually_exclusive_group()
    frontmost_group.add_argument(
        "--require-frontmost",
        dest="require_frontmost",
        action="store_true",
        help="Only send keystrokes when Warp is already frontmost (default).",
    )
    frontmost_group.add_argument(
        "--no-require-frontmost",
        dest="require_frontmost",
        action="store_false",
        help="Allow sending even if Warp is not frontmost (requires --allow-activate; unsafe otherwise).",
    )
    parser.set_defaults(require_frontmost=True)

    parser.add_argument(
        "--allow-activate",
        action="store_true",
        default=False,
        help="If set, activate Warp before sending (less safe).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Do not actually send 'Please continue'.",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    once_p = sub.add_parser("once", help="Run a single check/evaluate/send cycle")
    once_p.add_argument(
        "--print-debug",
        action="store_true",
        help="Print selected debug info to stderr.",
    )

    run_p = sub.add_parser("run", help="Run continuously")
    run_p.add_argument(
        "--poll-interval",
        type=float,
        default=float(os.environ.get("WARP_AUTOCONTINUE_POLL", "1.0")),
        help="Polling interval in seconds.",
    )

    return parser


def run_once_command(
    *, args: argparse.Namespace, db_path: Path, state_path: Path
) -> None:
    decide_and_maybe_continue(
        db_path=db_path,
        warp_bundle_id=args.warp_bundle_id,
        conversation_id=args.conversation_id,
        state_path=state_path,
        max_user_messages=args.max_user_messages,
        max_plan_chars=args.max_plan_chars,
        evaluator=args.evaluator,
        allow_activate=args.allow_activate,
        require_frontmost=args.require_frontmost,
        dry_run=args.dry_run,
    )

    if args.print_debug:
        st = load_state(state_path)
        eprint(json.dumps(st, indent=2))


def run_loop_command(
    *, args: argparse.Namespace, db_path: Path, state_path: Path
) -> None:
    interval = max(0.25, args.poll_interval)
    while True:
        try:
            decide_and_maybe_continue(
                db_path=db_path,
                warp_bundle_id=args.warp_bundle_id,
                conversation_id=args.conversation_id,
                state_path=state_path,
                max_user_messages=args.max_user_messages,
                max_plan_chars=args.max_plan_chars,
                evaluator=args.evaluator,
                allow_activate=args.allow_activate,
                require_frontmost=args.require_frontmost,
                dry_run=args.dry_run,
            )
        except KeyboardInterrupt:
            raise
        except Exception as e:
            eprint(f"warp-autocontinue: error: {e}")
        time.sleep(interval)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    db_path = locate_warp_db_path(args.db_path)
    state_path = Path(args.state_path).expanduser()

    if args.cmd == "once":
        run_once_command(args=args, db_path=db_path, state_path=state_path)
    elif args.cmd == "run":
        run_loop_command(args=args, db_path=db_path, state_path=state_path)


if __name__ == "__main__":
    main()
