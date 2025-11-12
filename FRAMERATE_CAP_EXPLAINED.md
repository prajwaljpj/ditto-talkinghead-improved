# Frame Rate Cap - Why and How

## TL;DR

**Do we need a frame rate cap?** YES, but not for the reasons you might think.

**Why cap at 40 FPS?**
1. **Bandwidth efficiency**: Higher FPS = more data transmitted over WebRTC
2. **Browser rendering**: Most browsers/displays are 60Hz max, anything above 60 FPS is wasted
3. **Network adaptation**: WebRTC needs consistent frame rates to optimize bitrate
4. **Resource management**: CPU/GPU for encoding and network transmission

**Current setup**:
- Ditto generates frames at **25 FPS** (controlled by audio feed rate)
- We cap at **40 FPS maximum** (drop frames if faster than 25ms intervals)
- This provides headroom for bursts while preventing waste

## Deep Dive: Frame Rate in WebRTC Streaming

### What Determines Frame Rate?

In our setup, the frame rate is determined by **multiple factors**:

```
Audio Feed Rate → Ditto Generation → Frame Callback → WebRTC Transmission → Browser Display
     5 Hz             25 FPS           Cap at 40         Adaptive           60Hz max
```

### 1. Source Frame Rate: 25 FPS

**Current Configuration** (webrtc/livekit_gemini_agent.py):

```python
# Silent audio generator feeds every 200ms
await asyncio.sleep(0.2)  # 5 Hz feed rate

# Each chunk generates 5 frames (chunksize[1]=5)
# Result: 5 Hz × 5 frames = 25 FPS
```

**This is controlled by**:
- Audio chunk feed rate (5 Hz = every 200ms)
- Frames per chunk (5 frames from chunksize=(3,5,2))
- **Natural constraint**: We feed audio at a fixed rate to Ditto

### 2. Frame Rate Cap: 40 FPS

**Why cap at 40 FPS when we generate at 25 FPS?**

```python
# webrtc/livekit_gemini_agent.py:390-398
min_interval = 0.025  # 25ms = 40 FPS maximum
if elapsed < min_interval:
    self._frames_dropped += 1
    return
```

**Reasons**:

#### A. Burst Protection
Ditto can theoretically generate frames faster than 25 FPS in bursts:
- StreamSDK has internal buffering
- Multiple worker threads can generate frames simultaneously
- During state transitions, frames might arrive in rapid succession

**Example scenario**:
```
t=0ms:    Frame 1 generated
t=10ms:   Frame 2 generated  ← Too fast! (only 10ms elapsed)
t=20ms:   Frame 3 generated  ← Too fast! (only 10ms elapsed)
t=40ms:   Frame 4 generated  ← OK (40ms from Frame 1)
```

Without a cap, we'd send frames at irregular intervals: 10ms, 10ms, 20ms, 40ms...
With 40 FPS cap: Drop frames 2 and 3, send frames at: 40ms, 40ms... (regular intervals)

#### B. WebRTC Needs Consistent Frame Times

**From WebRTC research**:
- WebRTC implements **congestion control** that adapts bitrate based on network conditions
- Varying frame rates cause **lagging issues** (framesReceived - framesDecoded buildup)
- Consistent frame intervals help WebRTC's adaptive bitrate algorithm work better

**What happens with inconsistent frame rates**:
```
Irregular:  [Frame] 10ms [Frame] 50ms [Frame] 5ms [Frame] 100ms
            ↓
WebRTC:     Can't predict next frame → Conservative bitrate allocation
            → Buffer buildup or starvation → Jitter/lag

Regular:    [Frame] 40ms [Frame] 40ms [Frame] 40ms [Frame] 40ms
            ↓
WebRTC:     Predictable pattern → Optimal bitrate allocation
            → Smooth playback → No jitter
```

#### C. Display Refresh Rate Limits

Most displays are **60Hz** (16.67ms per frame):
- 25 FPS = 40ms per frame ✅ (well within 60Hz)
- 40 FPS = 25ms per frame ✅ (still within 60Hz)
- 60 FPS = 16.67ms per frame ✅ (matches 60Hz)
- 120 FPS = 8.33ms per frame ❌ (wasted on 60Hz displays)

For **talking head avatars**, the human eye perceives smoothness at:
- 15 FPS: Visible stuttering
- 24 FPS: Cinema standard (barely smooth)
- 25 FPS: Smooth for most content ✅ **Our target**
- 30 FPS: Broadcast standard (very smooth)
- 60 FPS: Gaming standard (overkill for faces)

#### D. Bandwidth and Encoding Cost

**Video bitrate roughly scales with frame rate**:

| FPS | Relative Bandwidth | Use Case |
|-----|-------------------|----------|
| 15 FPS | 1x | Low bandwidth (very choppy) |
| 25 FPS | 1.67x | **Good balance** ✅ |
| 30 FPS | 2x | Broadcast standard |
| 60 FPS | 4x | High motion (overkill for faces) |

**For 1280×720 video**:
- 25 FPS @ 1 Mbps bitrate = ~40 KB per frame
- 60 FPS @ 2.4 Mbps bitrate = ~40 KB per frame (2.4× the bandwidth)

**Our use case** (talking head avatar):
- Low motion content (face movements are subtle)
- 25 FPS is sufficient for perceived smoothness
- No need for 60 FPS gaming-level responsiveness

## Why Not Use Exact 25 FPS Instead of 40 FPS Cap?

**Good question!** Why allow up to 40 FPS when we generate at 25 FPS?

### Option 1: Exact Cap at 25 FPS (33.33ms minimum)
```python
min_interval = 0.04  # 40ms = 25 FPS exactly
```

**Pros**:
- Matches our target frame rate exactly
- Maximum bandwidth efficiency

**Cons**:
- No tolerance for timing variations
- Might drop frames unnecessarily if they arrive at 39ms intervals
- More aggressive frame dropping

### Option 2: Cap at 40 FPS (25ms minimum) - **CURRENT**
```python
min_interval = 0.025  # 25ms = 40 FPS maximum
```

**Pros**:
- Provides headroom for burst protection
- Tolerates small timing variations (25-40ms still accepted)
- Still prevents excessive frame rates (>40 FPS)

**Cons**:
- Slightly less strict bandwidth control

### Option 3: No Cap
```python
# No frame dropping
```

**Pros**:
- All generated frames sent

**Cons**:
- Wasted bandwidth during bursts
- Irregular frame intervals confuse WebRTC
- Potential jitter/lag in browser

## Current Architecture Analysis

### Frame Rate Control Points

```
┌─────────────────────────────────────────────────────────────────┐
│  AUDIO FEED RATE (Primary Control)                             │
│  ════════════════════════════════════════════════════           │
│  await asyncio.sleep(0.2)  →  5 Hz  →  25 FPS (5 frames/chunk)│
│  ✅ This is the PRIMARY frame rate control                      │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  DITTO PROCESSING                                               │
│  ════════════════                                               │
│  StreamSDK generates frames at ~25 FPS (driven by audio)       │
│  Multi-threaded pipeline may cause burst generation            │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  FRAME RATE CAP (Secondary Control)                            │
│  ═════════════════════════════════════                         │
│  min_interval = 0.025  →  40 FPS max                           │
│  ✅ Prevents bursts, ensures consistent intervals               │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  WEBRTC TRANSMISSION                                            │
│  ═══════════════════                                            │
│  Adaptive bitrate based on network conditions                  │
│  Consistent frame intervals = better adaptation                │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  BROWSER RENDERING                                              │
│  ═════════════════                                              │
│  60Hz display refresh (16.67ms per frame)                      │
│  25-40 FPS fits comfortably within 60Hz budget                 │
└─────────────────────────────────────────────────────────────────┘
```

### What Would Happen Without the Cap?

**Scenario**: Ditto generates 100 frames in a burst during state transition

```
Without Cap:                     With 40 FPS Cap:
═══════════                      ═══════════════
All 100 frames sent              Drop 98 frames, send 2
↓                                ↓
WebRTC: "OMG! 100 frames!"       WebRTC: "Nice, 2 frames"
↓                                ↓
Bitrate spikes                   Consistent bitrate
↓                                ↓
Buffer bloat                     No buffer issues
↓                                ↓
Network congestion               Smooth transmission
↓                                ↓
Lag/jitter on client             Smooth playback
```

## Do We Need to Change Anything?

### Current Configuration Assessment

**Audio feed rate**: 5 Hz (every 200ms) → 25 FPS
- ✅ **Good**: Smooth animation for talking heads
- ✅ **Good**: Efficient bandwidth usage
- ✅ **Good**: Matches cinema/broadcast standards (24-30 FPS)

**Frame rate cap**: 40 FPS (25ms minimum interval)
- ✅ **Good**: Provides burst protection
- ✅ **Good**: Consistent for WebRTC
- ✅ **Good**: Well below display refresh limits

**Target FPS**: 30 FPS (stored but not actively used)
- ⚠️ **Misleading**: We actually generate 25 FPS, not 30 FPS
- ℹ️ **Note**: This is just a documentation variable

### Recommendations

#### Option A: Keep Current Setup (Recommended)
```python
# Audio feed rate
await asyncio.sleep(0.2)  # 5 Hz → 25 FPS

# Frame cap
min_interval = 0.025  # 40 FPS max
```

**Why**: Current setup is well-balanced for talking heads.

#### Option B: Increase to True 30 FPS
If you want exactly 30 FPS:

```python
# Audio feed rate (adjust for 30 FPS)
await asyncio.sleep(0.167)  # 6 Hz → 30 FPS (6 Hz × 5 frames)

# Frame cap (adjust to 45 FPS for headroom)
min_interval = 0.022  # 45 FPS max
```

**Trade-offs**:
- ✅ Slightly smoother (30 FPS vs 25 FPS)
- ❌ 20% more bandwidth
- ❌ 20% more GPU usage
- ⚠️ Marginal perceptual difference for talking heads

#### Option C: Remove Cap (Not Recommended)
```python
# Remove lines 390-398
# No frame dropping
```

**Why not**:
- ❌ Wasted bandwidth during bursts
- ❌ Inconsistent frame intervals
- ❌ Potential WebRTC adaptation issues

## Summary

### Why Do We Need a Frame Rate Cap?

1. **Primary reason**: Prevent bursts from overwhelming WebRTC
2. **Secondary reason**: Ensure consistent frame intervals for adaptive bitrate
3. **Tertiary reason**: Avoid wasting bandwidth on frames faster than display refresh

### Is Our Current Cap Correct?

**Yes!** The 40 FPS cap with 25 FPS generation is a **good balance**:

- **25 FPS generation**: Smooth enough for talking heads, efficient bandwidth
- **40 FPS cap**: Provides 15 FPS headroom for bursts, prevents excessive transmission
- **Headroom ratio**: 40/25 = 1.6× tolerance (reasonable buffer)

### What About "Exact" Frame Rate?

**You don't need exact frame rate control** because:

1. **WebRTC adapts**: It handles variable frame rates (within reason)
2. **Browser adapts**: Displays at 60Hz regardless of source FPS
3. **Human perception**: Can't distinguish 24-30 FPS for talking heads

**What you DO need**:
- ✅ **Consistent average frame rate** (we have: 25 FPS)
- ✅ **Burst protection** (we have: 40 FPS cap)
- ✅ **No long gaps** (we have: continuous audio feeding)

### Final Verdict

**Keep the current configuration.** It's well-designed for talking head streaming:
- 25 FPS is the sweet spot (smooth yet efficient)
- 40 FPS cap provides safety without waste
- No need for exact frame timing in WebRTC streaming

---

**Date**: 2025-11-11
**Topic**: Frame rate cap explanation
**Conclusion**: Current 40 FPS cap with 25 FPS generation is optimal
**Status**: ✅ No changes needed
