#!/usr/bin/env python3
"""
Test a specific Gemini model.
"""

import os
import sys

def test_model(model_name):
    """Test a specific model."""
    print("=" * 60)
    print(f"Testing Model: {model_name}")
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

    # Test the model
    print(f"2. Testing model: {model_name}")
    try:
        response = client.models.generate_content(
            model=model_name,
            contents="Say 'Hello, this model works!' and nothing else."
        )
        if response.text:
            print(f"   ✅ MODEL WORKS!")
            print(f"   Response: {response.text.strip()}")
            print()
            print("=" * 60)
            print("SUCCESS! You can use this model:")
            print(f"   export GEMINI_MODEL='{model_name}'")
            print("   ./start_gemini_agent.sh")
            print("=" * 60)
            return True
        else:
            print(f"   ⚠️  Model responded but no text")
            return False
    except Exception as e:
        error_msg = str(e)
        print(f"   ❌ FAILED: {error_msg}")
        print()

        if "not found" in error_msg.lower() or "not available" in error_msg.lower():
            print("This model is not available in your Vertex AI project.")
            print()
            print("Possible reasons:")
            print("1. Model name is incorrect")
            print("2. Model not available in region:", location)
            print("3. Model requires allowlist/early access")
            print()
            print("Working model found earlier: gemini-2.0-flash-exp")

        return False


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "gemini-2.5-flash-native-audio-preview-09-2025"

    try:
        success = test_model(model)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
