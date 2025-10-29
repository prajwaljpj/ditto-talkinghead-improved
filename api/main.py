"""
Main FastAPI application for Ditto Talking Head API.
"""

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from api.config import get_settings
from api.routers import health_router, streaming_router
from api.middleware import setup_cors, RequestLoggingMiddleware
from api.services import get_session_manager
from core.utils.logging_config import initialize_logging, get_logger

# Initialize logging system
initialize_logging()
logger = get_logger(__name__)

# Get settings
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("=" * 60)
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Backend: {settings.inference_backend}")
    logger.info(f"Config: {settings.cfg_pkl}")
    logger.info(f"Max concurrent sessions: {settings.max_concurrent_sessions}")
    logger.info("=" * 60)

    # Initialize session manager
    session_manager = get_session_manager(settings.max_concurrent_sessions)

    # Start background cleanup task
    cleanup_task = asyncio.create_task(periodic_cleanup(session_manager))

    logger.info("Application startup complete")

    yield

    # Shutdown
    logger.info("Shutting down application...")

    # Cancel cleanup task
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

    # Cleanup all sessions
    sessions = await session_manager.get_all_sessions()
    logger.info(f"Cleaning up {len(sessions)} active sessions")
    for session_id in list(sessions.keys()):
        await session_manager.remove_session(session_id)

    logger.info("Application shutdown complete")


async def periodic_cleanup(session_manager):
    """
    Periodic task to cleanup expired sessions.
    """
    while True:
        try:
            await asyncio.sleep(settings.cleanup_interval_seconds)
            cleaned = await session_manager.cleanup_expired_sessions(
                settings.session_timeout_seconds
            )
            if cleaned > 0:
                logger.info(f"Periodic cleanup: removed {cleaned} expired sessions")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.exception("Error in periodic cleanup task")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Real-time talking head synthesis API using Motion-Space Diffusion",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Setup middleware
setup_cors(app, settings)
app.add_middleware(RequestLoggingMiddleware)

# Include routers
app.include_router(health_router, prefix=settings.api_v1_prefix)
app.include_router(streaming_router, prefix=settings.api_v1_prefix)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "backend": settings.inference_backend,
        "docs": "/docs",
        "health": f"{settings.api_v1_prefix}/health"
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc),
            "path": str(request.url)
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower()
    )
