#!/usr/bin/env python3
"""
Simple test for Gemini Live API WebSocket connection.
"""

import asyncio
import os

async def test():
    from google import genai
    from google.genai import types

    project_id = os.getenv("VERTEX_PROJECT_ID", "gnani-video-ai")
    location = os.getenv("VERTEX_LOCATION", "us-central1")

    print(f"Project: {project_id}, Location: {location}")

    client = genai.Client(vertexai=True, project=project_id, location=location)

    model = "gemini-live-2.5-flash-preview-native-audio-09-2025"
    print(f"Testing model: {model}")

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
    )

    try:
        async with client.aio.live.connect(model=model, config=config) as session:
            print("✅ Connected successfully!")
            print("Model is available and Live API works!")
            return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test())
    exit(0 if result else 1)
