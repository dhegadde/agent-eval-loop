"""Client-side RPM throttle.

Gemini's free tier caps requests per minute per model (15 for flash-lite, 5 for
flash). LiteLLM's retry does not honour the server's `retryDelay` hint well, so
without a throttle you burn the whole minute's budget in a few seconds, then
stall on backoff.

This gates every LiteLLM call through a rolling-window semaphore. Slower, but it
finishes instead of dying halfway through an optimization run.

Import this BEFORE constructing any LiteLLM object.
"""

import threading
import time
from collections import deque

import litellm

_LOCK = threading.Lock()
_CALLS: deque = deque()


def install_rpm_limit(max_rpm: int = 12, window: float = 60.0) -> None:
    """Patch litellm.completion / acompletion to respect a rolling RPM cap.

    Set max_rpm slightly BELOW your quota (12 for a 15 RPM limit) - the window is
    rolling on Google's side and clock skew will otherwise trip it.
    """

    def _wait_for_slot():
        while True:
            with _LOCK:
                now = time.monotonic()
                while _CALLS and now - _CALLS[0] > window:
                    _CALLS.popleft()
                if len(_CALLS) < max_rpm:
                    _CALLS.append(now)
                    return
                sleep_for = window - (now - _CALLS[0]) + 0.1
            time.sleep(max(sleep_for, 0.1))

    orig_completion = litellm.completion
    orig_acompletion = litellm.acompletion

    def throttled_completion(*args, **kwargs):
        _wait_for_slot()
        return orig_completion(*args, **kwargs)

    async def throttled_acompletion(*args, **kwargs):
        _wait_for_slot()
        return await orig_acompletion(*args, **kwargs)

    litellm.completion = throttled_completion
    litellm.acompletion = throttled_acompletion