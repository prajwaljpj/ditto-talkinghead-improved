# Signaling Server Architecture - Complete Explanation

## Overview

The signaling server enables real-time talking head generation over WebRTC. Here's the complete flow:

```
┌─────────────┐                    ┌──────────────────┐
│   Browser   │  ←─── WebSocket ──→│ Signaling Server │
│   (Web UI)  │  ←──── WebRTC ────→│   (Python)       │
└─────────────┘                    └──────────────────┘
      ↑                                      ↓
      │                              ┌──────────────┐
      │                              │ Ditto Model  │
      │                              │  (TensorRT)  │
      └──────── Video + Audio ───────┴──────────────┘
```

## Phase 1: WebSocket Signaling (Setup)

### Step 1: Connection
```
Browser                          Server
  │                                │
  ├──── WebSocket Connect ────────→│
  │                                │
  ←──── Connection Accepted ───────┤
```

**Code:** `signaling_server.py` line 746
```python
async def handle_client(self, websocket: WebSocketServerProtocol):
    logger.info(f"New client connected: {websocket.remote_address}")
```

### Step 2: Initialization Request
```
Browser                          Server
  │                                │
  ├──── {"type": "connect",  ──────→│
  │      "source": "avatar.jpg"}   │
  │                                ├─ Initialize Ditto SDK
  │                                ├─ Create RTCPeerConnection
  │                                ├─ Setup video/audio tracks
  │                                │
  ←──── {"type": "ready"} ─────────┤
```

**Code:** `signaling_server.py` line 374-414
```python
async def handle_connect(self, message: dict):
    source = message.get("source")

    # Initialize Ditto
    await self.setup_ditto(source)

    # Create WebRTC peer connection
    self.pc = RTCPeerConnection()

    # Add video track (Ditto output)
    self.pc.addTrack(self.video_track)

    # Add audio track (for playback)
    self.buffered_audio_track = BufferedAudioTrack()
    self.pc.addTrack(self.buffered_audio_track)

    # Setup handler for incoming audio from browser
    @self.pc.on("track")
    async def on_track(track):
        if track.kind == "audio":
            # This is where we receive audio from browser
```

### Step 3: WebRTC Negotiation (SDP Exchange)
```
Browser                          Server
  │                                │
  ├──── {"type": "offer",  ────────→│
  │      "sdp": "..."}              │
  │                                ├─ Process offer
  │                                ├─ Create answer
  │                                │
  ←──── {"type": "answer",  ───────┤
        "sdp": "..."}               │
  │                                │
  ├──── ICE candidates ───────────→│
  ←──── ICE candidates ────────────┤
```

**What is SDP?**
Session Description Protocol - describes the media capabilities:
- Audio codecs (Opus, G.711, etc.)
- Video codecs (VP8, H.264, etc.)
- Network addresses
- Track directions (sendrecv, sendonly, recvonly)

**Code:** `signaling_server.py` line 620-652
```python
async def handle_offer(self, message: dict):
    offer = RTCSessionDescription(sdp=sdp, type="offer")
    await self.pc.setRemoteDescription(offer)

    answer = await self.pc.createAnswer()
    await self.pc.setLocalDescription(answer)

    await self.send_message({
        "type": "answer",
        "sdp": self.pc.localDescription.sdp
    })
```

## Phase 2: WebRTC Media Streams (Real-time Communication)

Once negotiation is complete, two WebRTC streams are established:

### Stream 1: Browser → Server (Audio Input)

```
Browser Microphone
      ↓
   [Capture] → [Encode: Opus] → [RTP Packets]
                                      ↓
                            [Network: UDP/ICE]
                                      ↓
Server receives:                [Decode: Opus]
                                      ↓
                            BufferedAudioTrack.recv()
                                      ↓
                           [Process & Store]
```

**Code Flow:**
```python
# 1. Browser sends audio via WebRTC
@self.pc.on("track")
async def on_track(track):
    if track.kind == "audio":
        while True:
            frame = await track.recv()  # ← Audio arrives here

            # 2. Convert to numpy
            audio_array = frame.to_ndarray()

            # 3. Normalize to float32 [-1, 1]
            audio_float = audio_array / 32768.0

            # 4. Keep 48kHz copy
            audio_48k = audio_float.copy()

            # 5. Downsample to 16kHz for Ditto
            audio_16k = scipy.signal.resample_poly(audio_float, up=1, down=3)

            # 6. Accumulate until we have 6400 samples (16kHz)
            audio_buffer = np.concatenate([audio_buffer, audio_16k])

            # 7. When buffer is full, send to Ditto
            if len(audio_buffer) >= 6400:
                chunk = audio_buffer[:6400]
                self.sdk.run_chunk(chunk, chunksize=(3, 5, 2))

                # 8. Store 48kHz audio for later playback
                # (synchronized with video frames)
```

### Stream 2: Server → Browser (Video + Audio Output)

```
Ditto Model generates frames
      ↓
DittoVideoTrack.recv()
      ↓
   [Encode: VP8/H.264] → [RTP Packets]
                              ↓
                    [Network: UDP/ICE]
                              ↓
Browser receives:      [Decode]
                              ↓
                       <video> element displays frame
```

**And simultaneously:**

```
BufferedAudioTrack (stored 48kHz audio)
      ↓
BufferedAudioTrack.recv()
      ↓
   [Encode: Opus] → [RTP Packets]
                         ↓
               [Network: UDP/ICE]
                         ↓
Browser receives:   [Decode: Opus]
                         ↓
                  <audio> plays sound
```

## Phase 3: Synchronization Mechanism

This is the critical part. Here's how audio and video stay in sync:

### Audio Flow (Detailed)

```
Browser microphone (48kHz, stereo)
      ↓
frame.to_ndarray() → shape (2, 960) for 20ms of stereo
      ↓
Convert to mono → 960 samples
      ↓
Normalize → audio_float (960 samples, 48kHz)
      ↓
      ├─→ Copy to audio_48k (960 samples) ─┐
      │                                      │
      └─→ Downsample to audio_16k (320)     │
                ↓                            │
         Accumulate in buffer                │
                ↓                            │
    Wait for 6400 samples (16kHz)           │
                ↓                            │
         Send to Ditto model                │
                ↓                            │
      [Model processes...]                  │
                                             │
Meanwhile, the 48kHz copy:                  │
                                             │
                                             ↓
                      Store in queue with timestamp
                      timestamp = accumulated_duration
                      accumulated_duration += (960 / 48000)
                                             ↓
                           audio_chunks_queue.append(
                               (chunk_48k, timestamp)
                           )
```

### Video Flow (Detailed)

```
Ditto model generates frame
      ↓
DittoVideoTrack.on_frame(frame_rgb, frame_idx, timestamp)
      ↓
Add to frame_queue
      ↓
DittoVideoTrack.recv() called by WebRTC
      ↓
Get frame from queue
      ↓
Pop corresponding audio chunk:
   audio_chunk, audio_timestamp = session.audio_chunks_queue.pop(0)
      ↓
Add audio to playback buffer:
   buffered_audio_track.add_audio_chunk(audio_chunk, audio_timestamp)
      ↓
Create video frame with matching PTS:
   video_frame.pts = int(audio_timestamp * 25)
      ↓
Return video frame to WebRTC
```

### PTS (Presentation Timestamp) Synchronization

Both audio and video frames have PTS values:

```
Audio:
  PTS = timestamp * 48000 (samples)
  time_base = 1/48000
  Playback time = PTS / 48000 = timestamp

Video:
  PTS = timestamp * 25 (frames)
  time_base = 1/25
  Playback time = PTS / 25 = timestamp
```

**Example:**
```
At timestamp = 2.0 seconds:

Audio:
  PTS = 2.0 * 48000 = 96000
  time_base = 1/48000
  Playback: 96000 / 48000 = 2.0s ✓

Video:
  PTS = 2.0 * 25 = 50
  time_base = 1/25
  Playback: 50 / 25 = 2.0s ✓

Both play at exactly 2.0 seconds!
```

## Key Classes

### 1. BufferedAudioTrack (lines 49-191)

**Purpose:** Sends audio back to browser

**Key methods:**
- `add_audio_chunk(chunk, timestamp)`: Store audio with timestamp
- `recv()`: Called by WebRTC, returns next audio frame

**Queue:** Stores `(av.AudioFrame, timestamp)` tuples

### 2. DittoVideoTrack (lines 193-310)

**Purpose:** Sends video frames to browser

**Key methods:**
- `on_frame(frame_rgb, frame_idx, timestamp)`: Callback from Ditto model
- `recv()`: Called by WebRTC, returns next video frame

**Queue:** Stores `(frame_rgb, timestamp, receive_time, frame_idx)` tuples

### 3. DittoWebRTCSession (lines 313-721)

**Purpose:** Manages one WebRTC session

**Contains:**
- `RTCPeerConnection`: WebRTC connection
- `DittoVideoTrack`: Video output
- `BufferedAudioTrack`: Audio output
- `StreamSDK`: Ditto model instance

**Queues:**
- `audio_chunks_queue`: Stores `(audio_48k, timestamp)` for syncing

## Message Protocol

### WebSocket Messages (JSON)

**Client → Server:**
```json
{"type": "connect", "source": "path/to/avatar.jpg"}
{"type": "offer", "sdp": "..."}
{"type": "ice-candidate", "candidate": {...}}
```

**Server → Client:**
```json
{"type": "ready", "message": "..."}
{"type": "answer", "sdp": "..."}
{"type": "ice-candidate", "candidate": {...}}
{"type": "error", "message": "..."}
```

### WebRTC Media Streams

**RTP (Real-time Transport Protocol):**
- Audio packets: Opus codec, 48kHz
- Video packets: VP8 or H.264 codec, 25 FPS
- Each packet has sequence number and timestamp
- Packets can arrive out of order (UDP)

## Timing and Synchronization

### Critical Timing Points

```
t=0.000s: User starts speaking
   ↓
t=0.020s: First audio frame arrives at server (20ms at 48kHz)
   ↓
t=0.020-0.400s: Accumulate audio frames
   ↓
t=0.400s: Have 6400 samples (16kHz), send to Ditto
   ↓
t=0.400-1.100s: Ditto processes (model latency ~700ms)
   ↓
t=1.100s: First video frame generated
   ↓
t=1.100s: Pop audio chunk with timestamp=0.000s
   ↓
t=1.100s: Send both to browser:
   - Video frame with PTS = 0.000 * 25 = 0
   - Audio frame with PTS = 0.000 * 48000 = 0
   ↓
Browser plays both at time 0.000s relative to stream start
```

## Where Things Can Go Wrong

### 1. Stereo Conversion
**Issue:** `flatten()` vs `mean()`
- Flatten: 960 stereo → 1920 values → wrong duration
- Mean: 960 stereo → 960 mono → correct duration

### 2. Timestamp Accumulation
**Issue:** If duration calculation is wrong, timestamps drift
```python
# Wrong:
chunk_duration = 1920 / 48000  # 40ms (but only have 20ms!)

# Right:
chunk_duration = 960 / 48000   # 20ms (correct)
```

### 3. PTS Calculation
**Issue:** Audio and video PTS must use same timestamp origin
```python
# Wrong:
audio.pts = self._timestamp  # Independent counter
video.pts = audio_timestamp * 25  # Different origin!

# Right:
audio.pts = timestamp * 48000  # Same origin
video.pts = timestamp * 25     # Same origin
```

### 4. WebRTC Buffering
**Issue:** Browser has jitter buffer (30-200ms)
- Can cause additional delay
- Varies by browser
- Network conditions affect it

### 5. Opus Encoding
**Issue:** Opus codec has frame sizes
- Prefers 20ms frames (960 samples @ 48kHz)
- Can use 10, 20, 40, or 60ms
- Wrong frame size → padding → quality issues

## Next Steps: Systematic Debugging

To find the actual issue, we need to check:

1. **What browser sends:**
   - Stereo or mono?
   - Sample count per frame?
   - Sample rate?

2. **What we process:**
   - After stereo conversion, how many samples?
   - What duration do we calculate?
   - What timestamps do we assign?

3. **What we send back:**
   - How many samples in audio frame?
   - What PTS values?
   - What do logs show?

4. **What browser plays:**
   - Need to check browser WebRTC internals
   - Check actual playback timing

Would you like me to create diagnostic tools to check each of these steps?
