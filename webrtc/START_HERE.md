# 🚀 How to Run Gemini + Ditto Conversational Avatar

## Step-by-Step Guide to Get Started

### Prerequisites Check

Before starting, make sure you have:

- ✅ Python 3.10 environment
- ✅ Ditto models downloaded and configured
- ✅ A Gemini API key (get from https://makersuite.google.com/app/apikey)
- ✅ An avatar image or video file

---

## 🎯 Option 1: Test the Pipeline (Recommended for First Time)

This tests the Gemini + Ditto integration without needing a browser.

### Step 1: Get Your Gemini API Key

1. Go to https://makersuite.google.com/app/apikey
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the key

### Step 2: Set the API Key

```bash
export GEMINI_API_KEY="your-api-key-here"
```

Or add to your shell config (~/.bashrc or ~/.zshrc):
```bash
echo 'export GEMINI_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

### Step 3: Verify Dependencies are Installed

```bash
# Make sure you're in the project root
cd /home/prajwaljpj/projects/talking_video/ditto-talkinghead

# Dependencies should already be installed from earlier
# If not, run:
uv sync
```

### Step 4: Find Your Files

You need these three things:

1. **Config pickle**: Usually `outputs/cfg_f_model.pkl` or similar
2. **Data root**: Usually `./` (current directory)
3. **Avatar source**: An image or video of the person

Check what you have:
```bash
# Look for config files
ls outputs/*.pkl

# Look for example avatars
ls examples/

# Or use your own image
ls path/to/your/avatar.jpg
```

### Step 5: Run the Test Server

```bash
python webrtc/gemini_server.py \
  --cfg_pkl outputs/cfg_f_model.pkl \
  --data_root ./ \
  --source examples/avatar.jpg
```

**Replace these paths** with your actual files!

### What You Should See:

```
======================================================================
Gemini + Ditto Conversational Avatar Server
======================================================================
Gemini Model: models/gemini-2.0-flash-exp
Voice: Puck
Ditto Config: outputs/cfg_f_model.pkl
Avatar Source: examples/avatar.jpg
Interruptions: Enabled
======================================================================

Initializing pipeline...
[GeminiLiveProcessor] Initialized with model: models/gemini-2.0-flash-exp, voice: Puck
[GeminiLiveProcessor] Connecting to Gemini Live API...
[GeminiLiveProcessor] Session initialized successfully
[DittoAvatarProcessor] Initialized with source: examples/avatar.jpg

Pipeline ready!

Note: This is a test runner for the Pipecat pipeline.
For full WebRTC functionality, use the signaling_server.py
which integrates this pipeline with WebRTC connections.

Press Ctrl+C to stop...
```

If you see this, **congratulations!** The pipeline is working.

---

## 🌐 Option 2: Full WebRTC with Browser (Currently Needs Integration)

**Note**: The current implementation provides the Pipecat pipeline. To use it with the browser UI, we need to integrate it with the existing `signaling_server.py`.

### Current State:
- ✅ GeminiProcessor created
- ✅ DittoProcessor updated
- ✅ Web UI created (gemini_chat.html)
- ⏳ Integration with signaling_server.py needed

### How to Integrate (For You to Do):

You have two options:

#### Option A: Modify Existing Signaling Server

Edit `webrtc/signaling_server.py` around line 370-494 (the audio handling section):

```python
# Current code (around line 370):
@self.pc.on("track")
async def on_track(track):
    if track.kind == "audio":
        # ... direct audio processing ...
        self.sdk.run_chunk(chunk, chunksize=(3, 5, 2))

# Change to:
from webrtc.processors.gemini_processor import GeminiLiveProcessor
from webrtc.processors.ditto_processor import DittoAvatarProcessor

gemini_proc = GeminiLiveProcessor(api_key=os.getenv("GEMINI_API_KEY"))
ditto_proc = DittoAvatarProcessor(cfg_pkl, data_root, source_path)

@self.pc.on("track")
async def on_track(track):
    if track.kind == "audio":
        # Create AudioRawFrame and pass to Gemini
        frame = AudioRawFrame(...)
        await gemini_proc.process_frame(frame, direction)
        # Gemini will output TTS or silence to Ditto
        # Ditto will output video frames
```

#### Option B: Create New Integrated Server (Simpler)

Create a new file that combines signaling + Gemini + Ditto:

```bash
# This would be: webrtc/gemini_webrtc_server.py
# Combines signaling_server.py with gemini_server.py
# TODO: Implementation needed
```

---

## 🧪 Quick Test Without WebRTC

Want to just test if things work? Try this simple script:

```bash
# Create a test script
cat > test_gemini.py << 'EOF'
import os
import asyncio
from webrtc.conversation_manager import ConversationManager, EmotionType

async def test():
    print("Testing Gemini integration...")

    # Check API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not set!")
        return

    print(f"✅ API key found: {api_key[:10]}...")

    # Test conversation manager
    conv = ConversationManager()

    # Add a test turn
    turn = await conv.add_turn(
        role="model",
        text="I'm so happy to help you today!",
    )

    print(f"✅ Detected emotion: {turn.emotion.value}")

    # Get context
    context = await conv.get_context()
    print(f"✅ Context has {len(context)} messages")

    summary = await conv.get_summary()
    print(f"✅ Conversation summary:")
    print(f"   Turns: {summary['turn_count']}")
    print(f"   Emotion: {summary['current_emotion']}")

    print("\n🎉 All tests passed!")

asyncio.run(test())
EOF

# Run it
python test_gemini.py
```

---

## 📋 Troubleshooting

### Error: "GEMINI_API_KEY not set"

```bash
# Make sure you exported it
export GEMINI_API_KEY="your-key"

# Verify it's set
echo $GEMINI_API_KEY
```

### Error: "No module named 'google.genai'"

```bash
# Dependencies not installed
uv sync

# Or manually:
pip install google-genai
```

### Error: "Config file not found"

```bash
# Check what config files you have
ls outputs/*.pkl

# Use the correct path
python webrtc/gemini_server.py \
  --cfg_pkl outputs/YOUR_ACTUAL_FILE.pkl \
  --data_root ./ \
  --source examples/avatar.jpg
```

### Error: "Source file not found"

```bash
# Check what images you have
ls examples/
ls .

# Use a valid image
python webrtc/gemini_server.py \
  --cfg_pkl outputs/cfg_f_model.pkl \
  --data_root ./ \
  --source YOUR_IMAGE.jpg
```

### Error: "ModuleNotFoundError: No module named 'stream_pipeline_online'"

```bash
# Make sure you're in the project root
cd /home/prajwaljpj/projects/talking_video/ditto-talkinghead

# Run from there
python webrtc/gemini_server.py ...
```

---

## 🎯 What Happens in Test Mode?

When you run `gemini_server.py`, it:

1. ✅ Initializes the Gemini Live API connection
2. ✅ Sets up the Ditto talking head pipeline
3. ✅ Creates the Pipecat processors
4. ⏳ Waits (but doesn't process actual audio without WebRTC)

**This is a test/validation mode** that verifies:
- API key works
- Gemini connection succeeds
- Ditto models load correctly
- Pipeline can be created

---

## 🚀 Next Steps After Testing

Once the test mode works:

### Option 1: Use Existing Signaling Server (Simpler)

```bash
# Run the existing signaling server
python webrtc/signaling_server.py \
  --cfg_pkl outputs/cfg_f_model.pkl \
  --data_root ./ \
  --source examples/avatar.jpg

# Open browser to:
# webrtc/client/web/index.html
```

**Note**: This uses the OLD audio pipeline (no Gemini).
To use Gemini, you need to modify signaling_server.py as described above.

### Option 2: Wait for Full Integration

I can help you integrate the Gemini pipeline with the signaling server if you want to proceed with that.

---

## 🎓 Understanding the Components

- **`gemini_server.py`**: Pipecat pipeline runner (test mode)
- **`signaling_server.py`**: WebRTC server (existing, no Gemini)
- **`gemini_processor.py`**: Gemini integration (created, ready to use)
- **`ditto_processor.py`**: Ditto integration (updated, ready to use)
- **`gemini_chat.html`**: Web UI (created, needs WebRTC server)

---

## 💡 Recommended Approach

**For First Time:**

1. ✅ Run `gemini_server.py` in test mode (verify it works)
2. ✅ Test the existing `signaling_server.py` (without Gemini)
3. ✅ Open `index.html` and verify WebRTC works
4. 🔧 Then integrate Gemini into signaling_server.py

**Quick Command to Start:**

```bash
# Terminal 1: Test Gemini pipeline
export GEMINI_API_KEY="your-key"
python webrtc/gemini_server.py \
  --cfg_pkl outputs/cfg_f_model.pkl \
  --data_root ./ \
  --source examples/avatar.jpg
```

---

## ❓ Need Help?

If you get stuck, let me know:
1. What command you ran
2. What error you see
3. What files you have in `outputs/` and `examples/`

I can help you debug and get it running!
