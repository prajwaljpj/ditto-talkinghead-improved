"""
Pydantic models for API requests.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class SessionStartRequest(BaseModel):
    """Request to start a new streaming session."""
    source_type: str = Field(..., description="Type of source: 'image' or 'video'")
    config: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional configuration parameters"
    )

    @field_validator('source_type')
    @classmethod
    def validate_source_type(cls, v):
        if v not in ['image', 'video']:
            raise ValueError('source_type must be "image" or "video"')
        return v


class SessionStopRequest(BaseModel):
    """Request to stop an active session."""
    session_id: str = Field(..., description="ID of the session to stop")


class StreamConfig(BaseModel):
    """Configuration for streaming inference."""
    # Avatar settings
    max_size: int = Field(default=1920, description="Maximum dimension for source")
    crop_scale: float = Field(default=2.3, description="Crop scale factor")
    crop_vx_ratio: float = Field(default=0.0, description="Horizontal crop offset ratio")
    crop_vy_ratio: float = Field(default=-0.125, description="Vertical crop offset ratio")

    # Emotion and control
    emo: int = Field(default=4, description="Emotion index (0-7)")

    # Audio2Motion settings
    sampling_timesteps: int = Field(default=50, description="Diffusion sampling timesteps")
    overlap_v2: int = Field(default=10, description="Audio overlap frames")

    # Output settings
    fps: int = Field(default=25, description="Output frames per second")

    # Advanced settings
    use_stitching: bool = Field(default=True, description="Enable motion stitching")
    relative_motion: bool = Field(default=True, description="Use relative motion")

    model_config = {"extra": "allow"}  # Allow additional fields


class AudioChunkMetadata(BaseModel):
    """Metadata for an audio chunk."""
    timestamp: float = Field(..., description="Timestamp in seconds")
    sample_rate: int = Field(default=16000, description="Audio sample rate")
    duration_ms: float = Field(..., description="Duration in milliseconds")
