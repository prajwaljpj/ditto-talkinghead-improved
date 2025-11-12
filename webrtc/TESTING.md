# WebRTC Testing Guide

This guide explains how to test the WebRTC pipeline with pre-recorded audio files for reproducible debugging.

## Overview

Testing with pre-recorded audio is much easier than using a live microphone because:
- **Reproducible**: Same audio every time
- **No browser needed**: Pure Python test client
- **Easy debugging**: Known input, can inspect output frame-by-frame
- **Automated testing**: Can be integrated into CI/CD

## Browser Recording Test (Recommended for Audio Debugging)

If you want to test audio quality in the browser with recording:

### 1. Start the server
```bash
uv run python webrtc/signaling_server.py \
    --cfg_pkl outputs/final_ckpt/cfg.pkl \
    --data_root outputs/final_ckpt
```

### 2. Start web server
```bash
cd webrtc/client/web
python3 -m http.server 8000
```

### 3. Open the recording test page
Open: **http://localhost:8000/record_test.html**

### 4. Test workflow:
1. Configure server URL: `ws://localhost:8080`
2. Set avatar path: `avatars/Hyperrealistic_Indian_Woman_Professional_Presentation_static_5_loop.jpg`
3. Click **"Start Recording"** 🎤
4. Speak clearly (count 1-10, read text, etc.)
5. Click **"Stop & Process"** ⏹️
6. Wait for video to process
7. Video plays automatically with your audio
8. **Listen carefully**: Is the audio clear? Any crackling/distortion?

This is the best way to debug audio quality issues!

---

## Python Test Client (For Automation)

### Quick Start

### 1. Generate Test Audio

```bash
# Generate 5-second speech-like audio
python webrtc/generate_test_audio.py --type speech --duration 5 --output test_audio.wav

# Or generate counting pattern (easier to verify lip sync)
python webrtc/generate_test_audio.py --type counting --output test_counting.wav

# Or use your own audio file (wav, mp3, etc.)
```

### 2. Start the Server

```bash
uv run python webrtc/signaling_server.py \
    --cfg_pkl outputs/final_ckpt/cfg.pkl \
    --data_root outputs/final_ckpt
```

### 3. Run Test Client

```bash
# Basic test (saves to current directory)
uv run python webrtc/test_client.py \
    --audio webrtc/test_audio.wav \
    --image example/image.png \
    --output test_output.mp4

# With custom duration
python webrtc/test_client.py \
    --audio test_audio.wav \
    --image path/to/your/avatar.jpg \
    --output test_output.mp4 \
    --duration 10

# Loop audio (for longer tests)
python webrtc/test_client.py \
    --audio test_audio.wav \
    --image path/to/avatar.jpg \
    --output test_output.mp4 \
    --loop
```

### 4. Inspect Output

The output file is saved in your **current working directory** (where you ran the command).

```bash
# Check if file was created
ls -lh test_output.mp4

# Play the output video
ffplay test_output.mp4

# Check video info
ffprobe test_output.mp4

# Extract frames for detailed inspection
mkdir frames
ffmpeg -i test_output.mp4 frames/frame_%04d.png
```

## Test Audio Types

### Speech Pattern (Default)
```bash
python webrtc/generate_test_audio.py --type speech --duration 5
```
- Simulates speech with varying frequencies (syllables)
- Good for testing overall lip sync behavior
- Has pauses between "words"

### Counting Pattern
```bash
python webrtc/generate_test_audio.py --type counting
```
- Distinct tones for numbers 1-10 with pauses
- Easy to verify timing: "number → lip movement → pause"
- Best for measuring latency

### Sine Wave
```bash
python webrtc/generate_test_audio.py --type sine --duration 3
```
- Simple continuous tone
- Good for testing audio quality (no crackling)
- Should produce smooth, consistent lip movement

## Using Real Audio Files

You can use any audio file (not just generated ones):

```bash
# Use your own recording
python webrtc/test_client.py \
    --audio my_recording.wav \
    --image avatar.jpg \
    --output result.mp4

# Use online TTS (requires internet + pyttsx3/gTTS)
# Example with gTTS:
from gtts import gTTS
tts = gTTS("Hello, I am testing the Ditto talking head system", lang='en')
tts.save("test_speech.mp3")

python webrtc/test_client.py \
    --audio test_speech.mp3 \
    --image avatar.jpg \
    --output result.mp4
```

## Debugging Tips

### Check Audio Normalization

Look at server logs for these lines:
```
INFO: First audio frame: format=s16, layout=mono, sample_rate=48000
INFO: First audio array: dtype=int16, shape=(960,), range=[-1234, 1567]
INFO: Storing frame 0: dtype=float32, range=[-0.0376, 0.0478], RMS=0.0234
INFO: First retrieved audio: dtype=float32, range=[-0.0376, 0.0478]
INFO: First int16 conversion: range=[-1232, 1566]
```

**Good signs:**
- Float32 range: **-1.0 to 1.0** ✓
- int16 range: **-32768 to 32767** ✓
- RMS values: **0.01 to 0.5** (depending on volume) ✓

**Bad signs:**
- Float32 range: `-30000 to 30000` ✗ (not normalized!)
- int16 range: `32767 to 32767` ✗ (clipping!)
- RMS values: `> 10.0` ✗ (way too loud)

### Check Audio-Video Sync

```bash
# Extract audio from output
ffmpeg -i test_output.mp4 -vn -acodec pcm_s16le output_audio.wav

# Compare input vs output audio
ffplay -f lavfi "amovie=test_audio.wav,asplit[a][b];[a]showwaves[wave1];[b]showwaves[wave2];[wave1][wave2]hstack"

# Check for delay
# Listen to both - output should be ~550ms delayed
```

### Check Frame Rate

```bash
ffprobe -v error -select_streams v -show_entries stream=r_frame_rate -of default=noprint_wrappers=1 test_output.mp4
```

Should show: `25/1` (25 fps)

### Check for Dropped Frames

Look in server logs:
```
WARNING: Frame queue full, dropping frame 123
WARNING: Audio buffer full, dropped 5 chunks total
```

If you see these, the pipeline is overloaded.

## Automated Testing

### Test Suite Example

```bash
#!/bin/bash
# test_suite.sh

echo "=== Ditto WebRTC Test Suite ==="

# Test 1: Short audio
echo "Test 1: Short audio (3s)"
python webrtc/generate_test_audio.py --type speech --duration 3 --output test1.wav
python webrtc/test_client.py --audio test1.wav --image test_avatar.jpg --output test1.mp4
[ -f test1.mp4 ] && echo "✓ Test 1 passed" || echo "✗ Test 1 failed"

# Test 2: Counting (timing test)
echo "Test 2: Counting pattern"
python webrtc/generate_test_audio.py --type counting --output test2.wav
python webrtc/test_client.py --audio test2.wav --image test_avatar.jpg --output test2.mp4
[ -f test2.mp4 ] && echo "✓ Test 2 passed" || echo "✗ Test 2 failed"

# Test 3: Long audio (stability test)
echo "Test 3: Long audio (15s)"
python webrtc/generate_test_audio.py --type speech --duration 15 --output test3.wav
python webrtc/test_client.py --audio test3.wav --image test_avatar.jpg --output test3.mp4
[ -f test3.mp4 ] && echo "✓ Test 3 passed" || echo "✗ Test 3 failed"

echo "=== Tests complete ==="
```

## Expected Results

### Latency
- **First frame**: ~900-1000ms (cold start)
- **Steady state**: ~550ms audio-to-video delay
- This is visible: speak → wait ~0.5s → lips move

### Audio Quality
- No crackling or distortion
- Clear echo of your input
- Volume similar to input

### Video Quality
- Smooth 25fps playback
- No stuttering or freezing
- Lip movements match audio timing (with ~550ms delay)

### Lip Sync Quality
Depends on model quality, not streaming:
- Timing should be consistent
- Mouth should open/close with speech
- Detailed phoneme matching may vary

## Troubleshooting

### Problem: No output file created

**Check:**
```bash
# Server logs - look for errors
# Client logs - connection successful?
```

**Solution:**
- Ensure server is running
- Check WebSocket URL is correct (ws://localhost:8080)
- Verify image path is correct

### Problem: Audio is crackling

**Check server logs for:**
```
First audio array: dtype=int16, range=[-30000, 30000]
First retrieved audio: range=[-30000, 30000]  ← Should be -1 to 1!
```

**Solution:** Audio normalization bug - check signaling_server.py lines 375-386

### Problem: Video is choppy

**Check:**
```bash
# GPU usage
nvidia-smi

# Server logs
Frame 25: Generated 2000.0ms ago  ← Too slow!
```

**Solution:**
- GPU overloaded - reduce concurrent connections
- Check TensorRT engines are compiled
- Reduce video resolution

### Problem: Audio-video out of sync

**Check server logs for:**
```
✓ Added 1 audio chunks for frame 0, buffer size: 0→1
⚠ Sending silence (count: 50), queue empty
```

**Solution:** Buffer underrun - should maintain 5+ chunks in buffer

## Comparison with Browser Testing

| Aspect | Test Client | Browser Client |
|--------|-------------|----------------|
| Setup | Just Python | Need HTTP server |
| Audio input | Pre-recorded file | Live microphone |
| Reproducibility | Perfect | Variable |
| Debugging | Easy (logs + output file) | Harder (console only) |
| Real-world | No | Yes |
| Automation | Easy | Requires Selenium |

**Recommendation:**
1. Use **test client** for development and debugging
2. Use **browser client** for final validation and demos

## Next Steps

After testing confirms the pipeline works:
1. Test with browser client for real-world validation
2. Measure actual latency with screen recording
3. Optimize if needed (see LIPSYNC_TESTING_GUIDE.md)
4. Deploy to production

---

For lip sync quality evaluation, see [LIPSYNC_TESTING_GUIDE.md](LIPSYNC_TESTING_GUIDE.md)
