"""
API models package.
"""

from .requests import (
    SessionStartRequest,
    SessionStopRequest,
    StreamConfig,
    AudioChunkMetadata,
)
from .responses import (
    SessionStatus,
    SessionResponse,
    HealthResponse,
    ModelInfo,
    ModelsResponse,
    ErrorResponse,
    WSInitMessage,
    WSAudioChunkMessage,
    WSStopMessage,
    WSStatusMessage,
    WSFrameMessage,
    WSProgressMessage,
    WSErrorMessage,
    WSCompleteMessage,
)

__all__ = [
    # Requests
    "SessionStartRequest",
    "SessionStopRequest",
    "StreamConfig",
    "AudioChunkMetadata",
    # Responses
    "SessionStatus",
    "SessionResponse",
    "HealthResponse",
    "ModelInfo",
    "ModelsResponse",
    "ErrorResponse",
    # WebSocket Messages
    "WSInitMessage",
    "WSAudioChunkMessage",
    "WSStopMessage",
    "WSStatusMessage",
    "WSFrameMessage",
    "WSProgressMessage",
    "WSErrorMessage",
    "WSCompleteMessage",
]
