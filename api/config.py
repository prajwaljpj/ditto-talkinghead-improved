"""
API Configuration management using Pydantic settings.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration settings.
    """
    # API Settings
    app_name: str = "Ditto Talking Head API"
    app_version: str = "1.0.0"
    api_v1_prefix: str = "/api/v1"
    debug: bool = False

    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1  # Keep at 1 for GPU inference
    reload: bool = False

    # CORS Settings
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]
    cors_allow_credentials: bool = True
    cors_allow_methods: list[str] = ["*"]
    cors_allow_headers: list[str] = ["*"]

    # Model Settings
    inference_backend: str = "tensorrt"  # "pytorch" or "tensorrt"
    cfg_pkl: str = "checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl"
    data_root: str = "checkpoints/ditto_trt_custom2"

    # GPU Settings
    gpu_id: int = 0
    max_concurrent_sessions: int = 2

    # Storage Settings
    upload_dir: Path = Path("api/uploads").resolve()
    output_dir: Path = Path("api/outputs").resolve()
    temp_dir: Path = Path("api/temp").resolve()
    max_upload_size_mb: int = 500

    # Session Settings
    session_timeout_seconds: int = 3600  # 1 hour
    cleanup_interval_seconds: int = 300  # 5 minutes

    # Logging Settings
    log_level: str = "INFO"
    log_dir: Path = Path("logs")
    enable_console_logging: bool = True
    enable_file_logging: bool = True

    # WebSocket Settings
    ws_heartbeat_interval: int = 30  # seconds
    ws_message_queue_size: int = 100

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_prefix="DITTO_"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories if they don't exist
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Adjust config based on backend
        if self.inference_backend == "tensorrt":
            self.cfg_pkl = "checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl"

    @property
    def tensorrt_available(self) -> bool:
        """Check if TensorRT backend is available."""
        trt_cfg = Path("checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl")
        return trt_cfg.exists()


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """
    Get application settings (FastAPI dependency).
    """
    return settings
