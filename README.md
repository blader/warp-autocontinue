# warp-autocontinue
A local “Option C” hook for Warp Desktop Agent sessions.

It monitors Warp’s on-disk state (`warp.sqlite`) and, when an agent run returns control to the user, it:
1. Extracts the last few user/assistant messages plus the in-context Plan (if any).
2. Uses an evaluator (AI if configured; heuristic fallback) to decide whether the agent is *done* or whether it’s just paused (progress update / question / error).
3. If it’s paused/incomplete, it automatically sends a user message: `Please continue`.

This is intentionally best-effort and may break across Warp updates.

## How it works (high level)
- Reads Warp’s local SQLite DB under `~/Library/Group Containers/2BBY89MBSN.dev.warp/Library/Application Support/.../warp.sqlite`.
- Pulls the most recently modified `agent_tasks.task` blob and extracts user/assistant messages by scanning for UUID-prefixed protobuf string fields.
- If a Plan exists, it’s stored as a notebook row (`notebooks.data`) keyed by `ai_document_id` and referenced from the task blob.
- If the evaluator says “continue”, it uses AppleScript (System Events) to type `Please continue` into Warp and press Return.

## Requirements
- macOS
- Python 3
- Warp (Stable or Preview)
- Accessibility permissions for `osascript`/System Events if you enable auto-send

## Installation
```bash
./install.sh
```

## Usage
Run continuously:
```bash
warp-autocontinue run
```

Run a single evaluation/decision cycle (no loop):
```bash
warp-autocontinue once
```

## Evaluator configuration
By default, this uses a heuristic evaluator (no external API).

To use OpenAI for evaluation:
```bash
export WARP_AUTOCONTINUE_EVAL=openai
export OPENAI_API_KEY=...   # required
export OPENAI_MODEL=gpt-4.1-mini  # optional
```

## Safety
By default, the tool only sends keystrokes when Warp is already the frontmost app (to avoid typing into the wrong window).
Set `--allow-activate` if you want it to activate Warp before sending.

## launchd
A template LaunchAgent plist is provided in `launchd/dev.warpception.warp-autocontinue.plist`.
