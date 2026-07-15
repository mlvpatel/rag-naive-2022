"""Rate limiting and input sanitization for RagFlow.

There is no authentication here on purpose. This baseline is a local reference
service: docker-compose binds it to the loopback interface, so it is reachable
from the host and nowhere else. A shipped default credential would be worse
than none, because it reads as protection while being public knowledge. Put a
real gateway in front of it before exposing it to a network.
"""

from __future__ import annotations

import bleach
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)


def sanitize(text: str) -> str:
    """Strip any HTML so a question cannot smuggle markup into the pipeline."""
    return bleach.clean(text or "", tags=[], strip=True).strip()
