"""
Session management for concurrent streaming sessions.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional
from dataclasses import dataclass, field

from core.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class StreamSession:
    """Represents an active streaming session."""
    session_id: str
    status: str = "initializing"
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    source_path: Optional[str] = None
    output_path: Optional[str] = None
    config: Dict = field(default_factory=dict)
    frames_processed: int = 0
    error_message: Optional[str] = None

    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = datetime.utcnow()

    def is_expired(self, timeout_seconds: int) -> bool:
        """Check if session has expired."""
        return (datetime.utcnow() - self.last_activity).total_seconds() > timeout_seconds


class SessionManager:
    """
    Manages streaming sessions with concurrency control.
    """

    def __init__(self, max_concurrent_sessions: int = 2):
        self.max_concurrent_sessions = max_concurrent_sessions
        self.sessions: Dict[str, StreamSession] = {}
        self._lock = asyncio.Lock()

        logger.info(f"SessionManager initialized with max_concurrent_sessions={max_concurrent_sessions}")

    async def create_session(self, config: Optional[Dict] = None) -> StreamSession:
        """
        Create a new streaming session.

        Args:
            config: Optional configuration dictionary

        Returns:
            StreamSession: Newly created session

        Raises:
            RuntimeError: If maximum concurrent sessions reached
        """
        async with self._lock:
            # Check if we can create a new session
            active_sessions = self.get_active_session_count()
            if active_sessions >= self.max_concurrent_sessions:
                logger.warning(
                    f"Max concurrent sessions reached: {active_sessions}/{self.max_concurrent_sessions}"
                )
                raise RuntimeError(
                    f"Maximum concurrent sessions ({self.max_concurrent_sessions}) reached. "
                    "Please try again later."
                )

            # Generate unique session ID
            session_id = str(uuid.uuid4())

            # Create session
            session = StreamSession(
                session_id=session_id,
                config=config or {},
                status="initializing"
            )

            self.sessions[session_id] = session

            logger.info(f"Session created: {session_id}", extra={
                'metadata': {
                    'session_id': session_id,
                    'active_sessions': active_sessions + 1,
                    'max_sessions': self.max_concurrent_sessions
                }
            })

            return session

    async def get_session(self, session_id: str) -> Optional[StreamSession]:
        """
        Get a session by ID.

        Args:
            session_id: Session identifier

        Returns:
            StreamSession or None if not found
        """
        return self.sessions.get(session_id)

    async def update_session_status(
        self,
        session_id: str,
        status: str,
        error_message: Optional[str] = None
    ) -> bool:
        """
        Update session status.

        Args:
            session_id: Session identifier
            status: New status
            error_message: Optional error message

        Returns:
            bool: True if updated successfully
        """
        session = await self.get_session(session_id)
        if not session:
            logger.warning(f"Attempted to update non-existent session: {session_id}")
            return False

        async with self._lock:
            session.status = status
            session.update_activity()
            if error_message:
                session.error_message = error_message

            logger.debug(f"Session {session_id} status updated to {status}")
            return True

    async def increment_frames(self, session_id: str, count: int = 1) -> None:
        """
        Increment frame count for a session.

        Args:
            session_id: Session identifier
            count: Number of frames to add
        """
        session = await self.get_session(session_id)
        if session:
            async with self._lock:
                session.frames_processed += count
                session.update_activity()

    async def remove_session(self, session_id: str) -> bool:
        """
        Remove a session.

        Args:
            session_id: Session identifier

        Returns:
            bool: True if removed successfully
        """
        async with self._lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                logger.info(f"Session removed: {session_id}", extra={
                    'metadata': {
                        'session_id': session_id,
                        'remaining_sessions': len(self.sessions)
                    }
                })
                return True
            return False

    def get_active_session_count(self) -> int:
        """
        Get the number of active sessions.

        Returns:
            int: Number of active sessions
        """
        active_statuses = ["initializing", "ready", "streaming"]
        return sum(1 for s in self.sessions.values() if s.status in active_statuses)

    async def cleanup_expired_sessions(self, timeout_seconds: int) -> int:
        """
        Remove expired sessions.

        Args:
            timeout_seconds: Timeout threshold in seconds

        Returns:
            int: Number of sessions cleaned up
        """
        async with self._lock:
            expired_sessions = [
                sid for sid, session in self.sessions.items()
                if session.is_expired(timeout_seconds) and session.status not in ["completed", "error"]
            ]

            for session_id in expired_sessions:
                logger.warning(f"Cleaning up expired session: {session_id}")
                del self.sessions[session_id]

            if expired_sessions:
                logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

            return len(expired_sessions)

    async def get_all_sessions(self) -> Dict[str, StreamSession]:
        """
        Get all sessions.

        Returns:
            Dict of all sessions
        """
        return self.sessions.copy()

    async def get_session_stats(self) -> Dict:
        """
        Get session statistics.

        Returns:
            Dict with session stats
        """
        total = len(self.sessions)
        active = self.get_active_session_count()

        status_counts = {}
        for session in self.sessions.values():
            status_counts[session.status] = status_counts.get(session.status, 0) + 1

        return {
            "total_sessions": total,
            "active_sessions": active,
            "max_concurrent": self.max_concurrent_sessions,
            "status_breakdown": status_counts
        }


# Global session manager instance
_session_manager: Optional[SessionManager] = None


def get_session_manager(max_concurrent: int = 2) -> SessionManager:
    """
    Get or create the global session manager instance.

    Args:
        max_concurrent: Maximum concurrent sessions

    Returns:
        SessionManager instance
    """
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager(max_concurrent_sessions=max_concurrent)
    return _session_manager
