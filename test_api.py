#!/usr/bin/env python3
"""
Simple test client for Ditto Talking Head API.

Tests:
1. Health check endpoint
2. GPU info endpoint
3. Models endpoint
4. WebSocket connection (without actual inference)

Usage:
    python test_api.py
    python test_api.py --url http://localhost:8000
"""

import argparse
import asyncio
import json
import sys
import requests
import websockets


def test_health(base_url: str):
    """Test the health endpoint."""
    print("\n" + "=" * 60)
    print("Testing Health Endpoint")
    print("=" * 60)

    try:
        response = requests.get(f"{base_url}/api/v1/health")
        response.raise_for_status()
        data = response.json()

        print(f"✓ Status: {data['status']}")
        print(f"✓ Version: {data['version']}")
        print(f"✓ Backend: {data['backend']}")
        print(f"✓ GPU Available: {data['gpu_available']}")
        print(f"✓ Active Sessions: {data['active_sessions']}")
        print(f"✓ Uptime: {data['uptime_seconds']}s")

        return True
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False


def test_gpu_info(base_url: str):
    """Test the GPU info endpoint."""
    print("\n" + "=" * 60)
    print("Testing GPU Info Endpoint")
    print("=" * 60)

    try:
        response = requests.get(f"{base_url}/api/v1/gpu-info")
        response.raise_for_status()
        data = response.json()

        if data.get("available"):
            print(f"✓ GPU Available: True")
            print(f"✓ Device Count: {data.get('device_count', 'N/A')}")
            if "gpus" in data:
                for gpu in data["gpus"]:
                    print(f"  - GPU {gpu['id']}: {gpu['name']}")
                    print(f"    Memory: {gpu['used_memory_mb']}/{gpu['total_memory_mb']} MB")
        else:
            print(f"✗ GPU Available: False")
            print(f"  Message: {data.get('message', 'N/A')}")

        return True
    except Exception as e:
        print(f"✗ GPU info failed: {e}")
        return False


def test_models(base_url: str):
    """Test the models endpoint."""
    print("\n" + "=" * 60)
    print("Testing Models Endpoint")
    print("=" * 60)

    try:
        response = requests.get(f"{base_url}/api/v1/models")
        response.raise_for_status()
        data = response.json()

        print(f"✓ Active Backend: {data['active_backend']}")
        print(f"✓ Available Models:")
        for model in data['models']:
            status = "LOADED" if model['loaded'] else "Available"
            print(f"  - {model['name']} ({model['backend']}) [{status}] on {model['device']}")

        return True
    except Exception as e:
        print(f"✗ Models endpoint failed: {e}")
        return False


async def test_websocket(base_url: str):
    """Test WebSocket connection."""
    print("\n" + "=" * 60)
    print("Testing WebSocket Connection")
    print("=" * 60)

    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
    ws_url = f"{ws_url}/api/v1/stream/ws"

    try:
        async with websockets.connect(ws_url) as websocket:
            # Receive initial status message
            message = await websocket.recv()
            data = json.loads(message)
            print(f"✓ Connected to WebSocket")
            print(f"✓ Initial message: {data.get('message', '')}")

            # Send a stop message (we're not doing actual inference)
            await websocket.send(json.dumps({"type": "stop"}))
            print(f"✓ Sent stop message")

            # Connection should close gracefully
            print(f"✓ WebSocket closed gracefully")

        return True
    except websockets.exceptions.ConnectionClosedOK:
        # This is expected when we send stop
        print(f"✓ WebSocket closed gracefully (OK)")
        return True
    except Exception as e:
        print(f"✗ WebSocket test failed: {e}")
        return False


def test_ready_endpoint(base_url: str):
    """Test the readiness endpoint."""
    print("\n" + "=" * 60)
    print("Testing Readiness Endpoint")
    print("=" * 60)

    try:
        response = requests.get(f"{base_url}/api/v1/ready")
        response.raise_for_status()
        data = response.json()

        if data.get("ready"):
            print(f"✓ Service is ready to accept requests")
        else:
            print(f"⚠ Service not ready: {data.get('reason', 'Unknown')}")

        return True
    except Exception as e:
        print(f"✗ Readiness check failed: {e}")
        return False


def test_stats_endpoint(base_url: str):
    """Test the streaming stats endpoint."""
    print("\n" + "=" * 60)
    print("Testing Streaming Stats Endpoint")
    print("=" * 60)

    try:
        response = requests.get(f"{base_url}/api/v1/stream/stats")
        response.raise_for_status()
        data = response.json()

        print(f"✓ Total Sessions: {data.get('total_sessions', 0)}")
        print(f"✓ Active Sessions: {data.get('active_sessions', 0)}")
        print(f"✓ Max Concurrent: {data.get('max_concurrent', 0)}")

        return True
    except Exception as e:
        print(f"✗ Stats endpoint failed: {e}")
        return False


async def main():
    parser = argparse.ArgumentParser(description="Test Ditto Talking Head API")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the API (default: http://localhost:8000)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Ditto Talking Head API Test Suite")
    print("=" * 60)
    print(f"Testing API at: {args.url}")

    results = []

    # Run tests
    results.append(("Health Check", test_health(args.url)))
    results.append(("GPU Info", test_gpu_info(args.url)))
    results.append(("Models List", test_models(args.url)))
    results.append(("Readiness Check", test_ready_endpoint(args.url)))
    results.append(("Streaming Stats", test_stats_endpoint(args.url)))
    results.append(("WebSocket Connection", await test_websocket(args.url)))

    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("✓ All tests passed!")
        sys.exit(0)
    else:
        print(f"✗ {total - passed} test(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
