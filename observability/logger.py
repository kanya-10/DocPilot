"""
Minimal structured logging for observability.

Every query and tool call gets logged as a JSON line with a timestamp, so
you can compute metrics (latency, tool usage rate, confidence distribution)
without a heavyweight tracing platform. Swap this for Langfuse/LangSmith by
changing only this file if you want a dashboard later.
"""

import json
import time
import sys

LOG_PATH = "docpilot_events.jsonl"


def log_event(event_type: str, **fields) -> None:
    record = {
        "timestamp": time.time(),
        "event": event_type,
        **fields,
    }
    line = json.dumps(record, default=str)
    try:
        with open(LOG_PATH, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass
    print(line, file=sys.stderr)
