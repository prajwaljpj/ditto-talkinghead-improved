#!/usr/bin/env python3
"""
Test Gemini Live API with native audio model.
"""

import asyncio
import os
import sys

async def test_live_api():
    """Test the Live API connection."""
    print("=" * 60)
    print("Testing Gemini Live API with Native Audio")
    print("=" * 60)
    print()

    # Check environment
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    project_id = os.getenv("VERTEX_PROJECT_ID")
    location = os.getenv("VERTEX_LOCATION", "us-central1")

    if not creds_path or not project_id:
        print("❌ Set GOOGLE_APPLICATION_CREDENTIALS and VERTEX_PROJECT_ID first")
        return False

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("❌ google-genai not installed")
        return False

    # Initialize client
    print("1. Initializing Vertex AI client...")
    try:
        client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location
        )
        print(f"   ✅ Connected to project: {project_id}")
        print(f"   ✅ Location: {location}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False
    print()

    # Test Live API connection
    print("2. Testing Live API WebSocket connection...")
    model_name = "gemini-live-2.5-flash-preview-native-audio-09-2025"
    print(f"   Model: {model_name}")
    print()

    try:
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
        )

        print("   Connecting to Live API...")
        async with client.aio.live.connect(
            model=model_name,
            config=config
        ) as session:
            print("   ✅ WebSocket connection established!")
            print()

            # Send a simple text message
            print("   Sending test message: 'Hello'")
            await session.send(
                {"text": "Say hello and introduce yourself in one sentence."}
            )

            # Receive response
            print("   Waiting for response...")
            audio_received = False
            text_received = ""

            async for response in session.receive():
                if response.server_content.model_turn:
                    for part in response.server_content.model_turn.parts:
                        if part.inline_data and part.inline_data.mime_type.startswith("audio/"):
                            audio_received = True
                            print(f"   ✅ Received audio chunk ({len(part.inline_data.data)} bytes)")
                        elif part.text:
                            text_received += part.text
                            print(f"   💬 Text: {part.text}")

                # Break after getting some response
                if audio_received or text_received:
                    break

            print()
            if audio_received:
                print("   ✅ Audio response received!")
            if text_received:
                print(f"   ✅ Text response: {text_received}")

        print()
        print("=" * 60)
        print("✅ LIVE API TEST PASSED!")
        print("=" * 60)
        print()
        print("The Live API is working correctly.")
        print("You can now run the agent:")
        print("  ./start_gemini_agent.sh")
        print()
        return True

    except Exception as e:
        error_msg = str(e)
        print(f"   ❌ Live API failed: {error_msg}")
        print()

        if "not found" in error_msg.lower() or "not available" in error_msg.lower():
            print("=" * 60)
            print("❌ MODEL NOT AVAILABLE")
            print("=" * 60)
            print()
            print("The native audio model is not available in your Vertex AI project.")
            print()
            print("Possible reasons:")
            print(f"1. Model not available in region: {location}")
            print("2. Model requires allowlist/early access")
            print("3. Try a different region (e.g., us-west1, europe-west1)")
            print()
            print("Alternative: Use gemini-2.0-flash-exp (confirmed working)")
            print("  export GEMINI_MODEL='gemini-2.0-flash-exp'")
            print("  ./start_gemini_agent.sh")

        elif "Permission" in error_msg or "denied" in error_msg:
            print("=" * 60)
            print("❌ PERMISSION ERROR")
            print("=" * 60)
            print()
            print("Your service account needs additional permissions.")
            print("Make sure it has: Vertex AI User role")

        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(test_live_api())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
