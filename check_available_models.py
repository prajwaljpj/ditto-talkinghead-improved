#!/usr/bin/env python3
"""
Check available Gemini models in Vertex AI.
"""

import os
import sys

def check_models():
    """Check available models."""
    print("=" * 60)
    print("Checking Available Gemini Models")
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

    # Test different model names
    print("2. Testing model names...")
    print()

    models_to_test = [
        "gemini-2.0-flash-exp",
        "gemini-live-2.5-flash-preview",
        "models/gemini-2.0-flash-exp",
        "models/gemini-live-2.5-flash-preview",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "models/gemini-1.5-flash",
        "models/gemini-1.5-pro",
    ]

    working_models = []

    for model_name in models_to_test:
        print(f"   Testing: {model_name}")
        try:
            response = client.models.generate_content(
                model=model_name,
                contents="Say 'OK' and nothing else."
            )
            if response.text:
                print(f"   ✅ WORKS: {model_name}")
                working_models.append(model_name)
            else:
                print(f"   ⚠️  No response from: {model_name}")
        except Exception as e:
            error_msg = str(e)
            if "not found" in error_msg.lower() or "not" in error_msg.lower():
                print(f"   ❌ NOT FOUND: {model_name}")
            else:
                print(f"   ❌ ERROR: {error_msg[:100]}")
        print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if working_models:
        print(f"✅ Found {len(working_models)} working model(s):")
        for model in working_models:
            print(f"   - {model}")
        print()
        print("Recommended for LiveKit agent:")
        print(f"   export GEMINI_MODEL='{working_models[0]}'")
    else:
        print("❌ No working models found")
        print()
        print("Try using Gemini API key instead:")
        print("1. Get API key from: https://aistudio.google.com/apikey")
        print("2. Unset Vertex AI variables:")
        print("   unset GOOGLE_APPLICATION_CREDENTIALS")
        print("   unset VERTEX_PROJECT_ID")
        print("3. Set API key:")
        print("   export GEMINI_API_KEY='your-api-key'")
        print("4. Run agent:")
        print("   ./start_gemini_agent.sh")

    print("=" * 60)

    return len(working_models) > 0


if __name__ == "__main__":
    try:
        success = check_models()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
