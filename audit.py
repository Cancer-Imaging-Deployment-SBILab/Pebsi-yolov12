from contextvars import ContextVar
from typing import Optional

# Stores the current user's ID for audit logging; set by upstream auth if available
current_user_context: ContextVar[Optional[str]] = ContextVar(
    "current_user_id", default=None
)


def set_current_user(user_id: Optional[str]) -> None:
    """Store the active user's ID in the request context."""
    current_user_context.set(user_id)


def get_current_user() -> Optional[str]:
    """Fetch the active user's ID from the request context."""
    return current_user_context.get(None)
