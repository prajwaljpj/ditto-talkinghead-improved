# StreamSDK Logging Guide

## Overview

The LiveKit Gemini Agent now includes comprehensive logging for StreamSDK (Ditto) operations. This helps you monitor the avatar generation pipeline and identify bottlenecks.

## Available Logs

### 1. Initialization Logs

**When:** During agent startup

**Logs:**
```
🔧 Initializing StreamSDK...
   Config: checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl
   Data root: checkpoints/ditto_trt_custom2/
   Source: avatars/Hyperrealistic_Indian_Woman_Professional_Presentation_static_5_loop.jpg
✅ StreamSDK created (X.XXms)
🔧 Setting up StreamSDK pipeline...
✅ StreamSDK setup complete (X.XXms)
   Online mode: True
   Streaming mode: True
```

**What it shows:**
- SDK initialization time
- Setup configuration
- Pipeline setup time
- Mode confirmation

### 2. Frame Generation Logs

**When:** Every time StreamSDK generates a frame

**First frame:**
```
🎬 First frame generated from StreamSDK: (720, 1280, 3)
   Frame index: 0, Timestamp: 0.000s
```

**Every 100 frames:**
```
📹 StreamSDK: Generated 100 frames (latest idx: 99)
```

**What it shows:**
- Frame dimensions
- Frame index and timestamp
- Total frames generated

### 3. Audio Chunk Processing Logs

**When:** Every time audio is fed to StreamSDK

**Real audio (from Gemini):**
```
🎤 StreamSDK: Feeding chunk to run_chunk (6480 samples, buffer size: 0)
✅ StreamSDK: run_chunk completed (15.23ms)
```

**Silent audio (idle state):**
```
🔇 StreamSDK: Feeding silent audio chunk (6480 samples)
✅ StreamSDK: Silent audio chunk processed (14.56ms)
```

**What it shows:**
- Chunk size being processed
- Buffer state
- Processing time

### 4. Queue Monitoring Logs

**When:** Every 10 chunks processed

**Logs:**
```
📊 StreamSDK queues: audio2motion_queue=2/100, motion_stitch_queue=1/100, warp_f3d_queue=0/100, decode_f3d_queue=0/100, putback_queue=0/100
```

**What it shows:**
- Current queue sizes
- Maximum queue sizes
- Queue utilization (helps identify bottlenecks)

**Warning when queues are full:**
```
⚠️  audio2motion_queue is 85.0% full - potential bottleneck!
```

### 5. Periodic Status Reports

**When:** Every 100 frames generated

**Logs:**
```
======================================================================
📊 STREAMSDK STATUS
======================================================================
  Frames generated: 100
  Frames dropped: 2
  Ditto chunks sent: 15
  Online mode: True
  Streaming mode: True
  Queue states:
    audio2motion_queue: 2/100 (2.0%)
    motion_stitch_queue: 1/100 (1.0%)
    warp_f3d_queue: 0/100 (0.0%)
    decode_f3d_queue: 0/100 (0.0%)
    putback_queue: 0/100 (0.0%)
======================================================================
```

**What it shows:**
- Overall statistics
- Queue health
- Worker thread status
- Potential issues

### 6. Final Status (on shutdown)

**When:** Agent is closing

**Logs:**
```
======================================================================
📊 STREAMSDK STATUS
======================================================================
  Frames generated: 1250
  Frames dropped: 15
  Ditto chunks sent: 187
  Online mode: True
  Streaming mode: True
  Queue states: ...
======================================================================
```

## StreamSDK Internal Print Statements

**Note:** StreamSDK uses `print()` statements internally, not Python logging. These will appear in your console output but won't be captured by the logger.

**Common StreamSDK print statements:**
```
==================== setup kwargs ====================
[Configuration details...]
==================================================
```

**To capture these:**
- They appear in stdout/stderr
- Can be redirected to a file: `python agent.py > logs.txt 2>&1`
- Or use a logging handler to capture stdout

## Log Levels

### INFO Level
- SDK initialization
- First frame generated
- Periodic status reports (every 100 frames)
- Final status on shutdown

### DEBUG Level
- Every chunk processing
- Queue status (every 10 chunks)
- Frame generation details
- Profiling information

### WARNING Level
- Queue nearly full (>80%)
- Worker thread exceptions
- Frame delays

### ERROR Level
- Worker thread exceptions
- SDK initialization failures
- SDK close errors

## Enabling More Detailed Logs

### Option 1: Set Log Level to DEBUG
```python
logging.basicConfig(level=logging.DEBUG)
```

Or via environment:
```bash
export PYTHONUNBUFFERED=1  # For real-time output
python livekit_gemini_agent.py dev
```

### Option 2: Enable Profiling
```bash
export ENABLE_PROFILING=true  # Default: true
```

This adds detailed timing information for each stage.

### Option 3: Capture StreamSDK Print Statements
```python
import sys
import logging

class PrintToLogger:
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
    
    def write(self, message):
        if message.strip():
            self.logger.log(self.level, message.strip())
    
    def flush(self):
        pass

# Redirect stdout to logger
sys.stdout = PrintToLogger(logger, logging.INFO)
```

## Understanding Queue States

### Healthy State
```
audio2motion_queue: 2/100 (2.0%)
motion_stitch_queue: 1/100 (1.0%)
warp_f3d_queue: 0/100 (0.0%)
decode_f3d_queue: 0/100 (0.0%)
putback_queue: 0/100 (0.0%)
```
- Queues are mostly empty
- Processing is keeping up
- No bottlenecks

### Bottleneck Warning
```
⚠️  audio2motion_queue is 85.0% full - potential bottleneck!
```
- Queue is filling up faster than it's being processed
- May indicate:
  - GPU is overloaded
  - Model inference is slow
  - Too much audio being fed

### Queue Full (Critical)
```
audio2motion_queue: 100/100 (100.0%)
```
- Queue is completely full
- New chunks will block or be dropped
- Immediate action needed:
  - Reduce audio input rate
  - Check GPU utilization
  - Check for model errors

## StreamSDK Pipeline Stages

The StreamSDK processes audio through these stages (each has a queue):

1. **audio2motion_queue**: Audio features → Motion generation
2. **motion_stitch_queue**: Motion stitching
3. **warp_f3d_queue**: 3D warping
4. **decode_f3d_queue**: Frame decoding
5. **putback_queue**: Final frame composition

**Monitoring tip:** If `audio2motion_queue` fills up, the bottleneck is in motion generation. If later queues fill up, the bottleneck is downstream.

## Example Log Output

```
INFO:__mp_main__:🔧 Initializing StreamSDK...
INFO:__mp_main__:   Config: checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl
INFO:__mp_main__:✅ StreamSDK created (1234.56ms)
INFO:__mp_main__:🔧 Setting up StreamSDK pipeline...
INFO:__mp_main__:✅ StreamSDK setup complete (5678.90ms)
INFO:__mp_main__:   Online mode: True
INFO:__mp_main__:   Streaming mode: True
DEBUG:__mp_main__:🎤 StreamSDK: Feeding chunk to run_chunk (6480 samples, buffer size: 0)
DEBUG:__mp_main__:✅ StreamSDK: run_chunk completed (15.23ms)
INFO:__mp_main__:🎬 First frame generated from StreamSDK: (720, 1280, 3)
INFO:__mp_main__:   Frame index: 0, Timestamp: 0.000s
DEBUG:__mp_main__:📊 StreamSDK queues: audio2motion_queue=1/100, motion_stitch_queue=0/100, ...
INFO:__mp_main__:📊 Frames: 100 sent, 2 dropped
INFO:__mp_main__:======================================================================
INFO:__mp_main__:📊 STREAMSDK STATUS
INFO:__mp_main__:======================================================================
INFO:__mp_main__:  Frames generated: 100
INFO:__mp_main__:  Frames dropped: 2
INFO:__mp_main__:  Ditto chunks sent: 15
INFO:__mp_main__:  Online mode: True
INFO:__mp_main__:  Streaming mode: True
INFO:__mp_main__:  Queue states:
INFO:__mp_main__:    audio2motion_queue: 2/100 (2.0%)
INFO:__mp_main__:    motion_stitch_queue: 1/100 (1.0%)
INFO:__mp_main__:======================================================================
```

## Troubleshooting

### No frames being generated
- Check: `🎬 First frame generated` log
- If missing: StreamSDK may not be receiving audio
- Check queue states: Are queues empty or full?

### Frames dropping
- Check: `Frames dropped` count
- High drop rate: Frame generation is too fast
- Solution: Adjust frame pacing or reduce input rate

### Queues filling up
- Check: Queue status logs
- If `audio2motion_queue` fills: Motion generation bottleneck
- If later queues fill: Downstream processing bottleneck
- Solution: Check GPU utilization, reduce input rate

### Slow frame generation
- Check: Profiling logs for `ditto_chunk` timing
- High times (>50ms): Model inference is slow
- Check: GPU utilization and temperature

## Summary

You now have comprehensive visibility into:
- ✅ StreamSDK initialization and setup
- ✅ Frame generation progress
- ✅ Audio chunk processing
- ✅ Internal queue states
- ✅ Bottleneck detection
- ✅ Performance statistics

All logs are integrated with the agent's logging system and respect log levels.

