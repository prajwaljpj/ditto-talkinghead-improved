"""
Health check and status endpoints.
"""

import time
import torch
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from api.config import get_settings, Settings
from api.models import HealthResponse, ModelsResponse, ModelInfo
from api.services import get_session_manager, get_inference_service
from core.utils.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["health"])

# Track startup time
_startup_time = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check(settings: Settings = Depends(get_settings)):
    """
    Health check endpoint.

    Returns service status, GPU availability, and active sessions.
    """
    # Check GPU availability
    gpu_available = torch.cuda.is_available()

    # Get session stats
    session_manager = get_session_manager(settings.max_concurrent_sessions)
    active_sessions = session_manager.get_active_session_count()

    # Calculate uptime
    uptime = time.time() - _startup_time

    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        backend=settings.inference_backend,
        gpu_available=gpu_available,
        active_sessions=active_sessions,
        uptime_seconds=round(uptime, 2)
    )


@router.get("/models", response_model=ModelsResponse)
async def list_models(settings: Settings = Depends(get_settings)):
    """
    List available models and backends.
    """
    models = []

    # PyTorch backend
    pytorch_cfg = "checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl"
    models.append(ModelInfo(
        name="Ditto v0.4 (PyTorch)",
        backend="pytorch",
        loaded=(settings.inference_backend == "pytorch"),
        device="cuda" if torch.cuda.is_available() else "cpu"
    ))

    # TensorRT backend
    if settings.tensorrt_available:
        models.append(ModelInfo(
            name="Ditto v0.4 (TensorRT)",
            backend="tensorrt",
            loaded=(settings.inference_backend == "tensorrt"),
            device="cuda"
        ))

    return ModelsResponse(
        models=models,
        active_backend=settings.inference_backend
    )


@router.get("/gpu-info")
async def gpu_info():
    """
    Get detailed GPU information.
    """
    if not torch.cuda.is_available():
        return {
            "available": False,
            "message": "No CUDA-capable GPU detected"
        }

    try:
        import pynvml
        pynvml.nvmlInit()

        device_count = torch.cuda.device_count()
        gpus = []

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            gpus.append({
                "id": i,
                "name": name,
                "total_memory_mb": memory_info.total // (1024 * 1024),
                "used_memory_mb": memory_info.used // (1024 * 1024),
                "free_memory_mb": memory_info.free // (1024 * 1024),
                "utilization": pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            })

        pynvml.nvmlShutdown()

        return {
            "available": True,
            "device_count": device_count,
            "gpus": gpus
        }

    except Exception as e:
        logger.warning(f"Error getting GPU info: {e}")
        return {
            "available": True,
            "device_count": torch.cuda.device_count(),
            "error": str(e)
        }


@router.get("/ready")
async def readiness_check(settings: Settings = Depends(get_settings)):
    """
    Readiness check for Kubernetes/Docker health probes.

    Returns 200 if service is ready to accept requests.
    """
    # Check if GPU is required and available
    if settings.inference_backend == "tensorrt" and not torch.cuda.is_available():
        return {"ready": False, "reason": "TensorRT backend requires GPU"}

    # Check if we can accept new sessions
    session_manager = get_session_manager(settings.max_concurrent_sessions)
    active_sessions = session_manager.get_active_session_count()

    if active_sessions >= settings.max_concurrent_sessions:
        return {"ready": False, "reason": "Max concurrent sessions reached"}

    return {"ready": True}


@router.get("/live")
async def liveness_check():
    """
    Liveness check for Kubernetes/Docker health probes.

    Always returns 200 if the service is running.
    """
    return {"alive": True}


@router.get("/outputs/{filename}")
async def download_output(filename: str, settings: Settings = Depends(get_settings)):
    """
    Download generated output video files.

    Args:
        filename: Name of the output file (e.g., session_id_output.mp4)
    """
    logger.info(f"Download request for file: {filename}")
    logger.info(f"Output directory: {settings.output_dir}")

    file_path = settings.output_dir / filename

    # Check if final file exists (with audio), otherwise fall back to .tmp.mp4
    if not file_path.exists():
        tmp_file_path = settings.output_dir / f"{filename}.tmp.mp4"
        if tmp_file_path.exists():
            logger.info(f"Final file not found, using temp file: {tmp_file_path}")
            file_path = tmp_file_path
            filename = f"{filename}.tmp.mp4"
        else:
            logger.warning(f"Neither final file nor temp file found for: {filename}")

    logger.info(f"Full file path: {file_path}")
    logger.info(f"File exists: {file_path.exists()}")

    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    if not file_path.is_file():
        logger.error(f"Path exists but is not a file: {file_path}")
        raise HTTPException(status_code=400, detail="Invalid file path")

    # Security: Ensure file is within output directory
    try:
        file_path.resolve().relative_to(settings.output_dir.resolve())
    except ValueError:
        logger.error(f"Security violation: file outside output directory")
        raise HTTPException(status_code=403, detail="Access denied")

    logger.info(f"Serving file: {file_path}")
    return FileResponse(
        path=str(file_path),
        media_type="video/mp4",
        filename=filename
    )
