#!/usr/bin/env python3
"""
Startup script for Ditto Talking Head API.

Usage:
    python start_api.py
    python start_api.py --reload  # For development with hot reload
    python start_api.py --host 0.0.0.0 --port 8080
"""

import argparse
import uvicorn

from api.config import settings


def main():
    parser = argparse.ArgumentParser(description="Start Ditto Talking Head API")
    parser.add_argument(
        "--host",
        type=str,
        default=settings.host,
        help=f"Host to bind (default: {settings.host})"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.port,
        help=f"Port to bind (default: {settings.port})"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1, keep at 1 for GPU)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=settings.log_level.lower(),
        choices=["debug", "info", "warning", "error", "critical"],
        help="Log level"
    )

    args = parser.parse_args()

    print("=" * 60)
    print(f"Starting {settings.app_name} v{settings.app_version}")
    print("=" * 60)
    print(f"Backend: {settings.inference_backend}")
    print(f"Config: {settings.cfg_pkl}")
    print(f"Data root: {settings.data_root}")
    print(f"Host: {args.host}:{args.port}")
    print(f"Max concurrent sessions: {settings.max_concurrent_sessions}")
    print(f"Docs: http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}/docs")
    print("=" * 60)

    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level=args.log_level,
        access_log=True
    )


if __name__ == "__main__":
    main()
