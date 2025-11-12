# Lip Sync Testing and Evaluation Guide

This guide helps you evaluate and improve the lip sync quality of your Ditto WebRTC streaming setup.

## Understanding Lip Sync Quality

Good lip sync means the avatar's mouth movements match the timing and shape of the spoken words. Key aspects:

1. **Timing Accuracy**: Lips move at the right time relative to sound
2. **Phoneme Matching**: Mouth shapes correspond to speech sounds (vowels, consonants)
3. **Movement Smoothness**: Natural transitions between mouth positions
4. **Latency**: Acceptable delay between speaking and seeing lip movement

## Expected Latency

Your current unoptimized setup has these latency components:

| Component | Latency | Notes |
|-----------|---------|-------|
| Audio buffering | 400ms | Accumulating 6400 samples (10 frames) |
| HuBERT processing | 40ms | Audio feature extraction |
| LMDM inference | ~500ms | First batch (cold start) |
| LMDM inference | ~7ms | Subsequent batches (amortized) |
| Motion processing | 30ms | MotionStitch + WarpF3D + DecodeF3D |
| Network round-trip | 60-100ms | Client ↔ Server |
| Client decode/render | 20-30ms | Browser video decode |
| **Total (cold start)** | **~1050ms** | First response |
| **Total (steady state)** | **~550ms** | Ongoing conversation |

**Note**: 550ms (~0.5 seconds) is noticeable but acceptable for many use cases. For comparison:
- Video calls typically have 150-300ms latency
- Lip-dubbed videos need <100ms for "perfect" sync
- Speech-to-animation in games: 200-500ms is common

## Testing Procedures

### Test 1: Simple Vowel Sounds

**Purpose**: Check basic phoneme recognition and timing

1. Start the connection and enable audio
2. Speak each vowel sound clearly with 1-second pauses:
   - "AH" (as in "father") - Mouth should open wide
   - "EE" (as in "see") - Mouth should widen horizontally
   - "OO" (as in "food") - Lips should round/pucker
   - "OH" (as in "go") - Mouth should form oval shape

**Expected result**: You should see distinct mouth shapes for each vowel, appearing ~550ms after you speak.

**If it fails**:
- Random movements: Audio might not be reaching the SDK properly
- No movement: Check audio levels in browser (should show activity)
- Wrong shapes: May be model quality issue, not streaming issue

### Test 2: Counting with Pauses

**Purpose**: Check timing consistency and idle animation

1. Count slowly from 1 to 10 with 2-second pauses between numbers
2. Watch for:
   - Lips moving when you speak each number
   - Lips returning to rest position during pauses
   - Consistent delay between speaking and movement

**Expected result**: Each number should trigger lip movement ~550ms after you speak, then return to neutral.

**If it fails**:
- Increasing delay: Check server logs for queue buildup
- No pauses in movement: Audio chunks may be overlapping
- Erratic timing: Check network connection quality

### Test 3: Continuous Speech

**Purpose**: Test sustained conversation and motion smoothness

1. Read a paragraph of text continuously (10-15 seconds)
2. Watch for:
   - Smooth transitions between mouth positions
   - No sudden jumps or freezes
   - Consistent frame rate (~25fps)

**Expected result**: Smooth, continuous lip movement throughout speech.

**If it fails**:
- Stuttering: Check frame generation rate in logs
- Freezing: May indicate GPU overload or queue overflow
- Choppy network: Check WebRTC stats in browser console

### Test 4: Bilabial Consonants

**Purpose**: Check detailed phoneme matching (if model supports)

Speak words with clear "B", "M", "P" sounds (lips should close):
- "Baby"
- "Mama"
- "Papa"
- "Bumble bee"

**Expected result**: Lips should close for B/M/P sounds, appearing ~550ms after speaking.

**Note**: This depends on model quality. Some models may not capture fine consonant details.

## Diagnostic Tools

### 1. Server Logs

The updated server now logs:

```
INFO:__main__:Sending audio chunk #1: 6400 samples, RMS: 0.2341, range: [-0.8123, 0.7654]
INFO:__main__:First frame generated at frame_idx=0
INFO:__main__:Frame 25: Generated 45.3ms ago, Queue size: 2
```

**What to look for**:
- **Audio chunk RMS**: Should be 0.1-0.5 when speaking, <0.05 during silence
- **Audio chunk rate**: Should send every ~400ms during continuous speech
- **Frame generation timing**: Should be 40-50ms typically
- **Queue size**: Should be 1-3; if >5, frames are backing up

### 2. Browser Console

Check WebRTC stats by opening Developer Tools (F12) → Console:

```javascript
// Get stats
const stats = await pc.getStats();
stats.forEach(stat => {
    if (stat.type === 'inbound-rtp' && stat.kind === 'video') {
        console.log('FPS:', stat.framesPerSecond);
        console.log('Frames received:', stat.framesReceived);
        console.log('Frames dropped:', stat.framesDropped);
    }
});
```

**What to look for**:
- **FPS**: Should be ~25 fps
- **Frames dropped**: Should be 0 or very low (<1%)
- **Jitter**: Should be <50ms

### 3. Visual Latency Test

To measure actual perceived latency:

1. Record your screen while testing
2. Say a distinct word (like "NOW") clearly
3. Review the recording frame-by-frame to measure delay

**Tip**: Use a simple phrase like "one, two, three" and count frames between audio and lip movement.

## Common Issues and Solutions

### Issue: Lips move randomly, not matching speech

**Possible causes**:
1. Audio not being received properly
2. Audio format/normalization issues
3. Model not properly initialized

**Diagnosis**:
- Check server logs for audio chunk messages
- Verify RMS values are >0.1 when speaking
- Look for errors in SDK initialization

**Solutions**:
- Ensure microphone is selected and enabled
- Check audio constraints in `app.js` (sampleRate: 16000)
- Restart server and client connection

### Issue: Long delay (>1 second) between speech and lip movement

**Possible causes**:
1. Network latency
2. Queue buildup
3. GPU overload

**Diagnosis**:
- Check frame queue size in logs (should be <5)
- Monitor GPU usage (nvidia-smi)
- Check network round-trip time

**Solutions**:
- Reduce `min_chunk_size` to 3240 (reduces audio buffer to 200ms)
- Optimize LMDM inference (see README optimization section)
- Use better network connection

### Issue: Choppy or stuttering video

**Possible causes**:
1. Frame generation too slow
2. Network packet loss
3. Client-side decode issues

**Diagnosis**:
- Check "Frame X: Generated Yms ago" - should be <100ms
- Check browser console for dropped frames
- Monitor CPU usage on client

**Solutions**:
- Ensure server has GPU (not running on CPU)
- Check TURN/STUN configuration for NAT traversal
- Try different video quality settings in WebRTC

### Issue: Mouth shapes don't match phonemes

**Possible causes**:
1. Model limitation (not trained for detailed phonemes)
2. Audio2Motion model quality
3. Wrong emotion setting affecting expressions

**Note**: This is typically a MODEL issue, not a streaming issue. The Ditto model is trained on talking head datasets which may not capture all phoneme details.

**Diagnosis**:
- Test with different avatar sources
- Try different emotion settings (--emo parameter)
- Check if issue is consistent across different words

**Solutions**:
- Use higher quality source images (clear face, good lighting)
- Experiment with emotion settings (neutral=4 often works best)
- Consider fine-tuning the model on your specific avatar (advanced)

## Model Quality vs. Streaming Quality

**It's important to distinguish**:

1. **Streaming issues**: Timing, latency, stuttering, network problems
   - These can be fixed by optimizing the streaming pipeline

2. **Model issues**: Poor phoneme matching, unnatural expressions, limited motion
   - These require model retraining or higher-quality models

If the timing is correct but the mouth shapes don't look right, it's likely a model quality issue, not a streaming problem.

## Optimization Recommendations

If lip sync timing is acceptable but you want to reduce latency:

1. **Reduce audio buffering**: Change `min_chunk_size` from 6400 to 3240
   - Reduces latency by ~200ms
   - May affect motion quality (fewer context frames)

2. **Optimize LMDM inference**:
   - Use FP16 precision (requires model conversion)
   - Reduce LMDM sampling steps
   - Use TensorRT optimizations

3. **Pipeline parallelization**:
   - Run Audio2Motion and rendering in parallel
   - Pre-generate idle frames during silence

4. **Hardware encoding**:
   - Use NVENC for H.264 encoding
   - Offloads CPU and reduces latency

See the main README for detailed optimization guides.

## Expected Results

With the current unoptimized setup:

✅ **Good**:
- Consistent ~550ms latency (noticeable but acceptable)
- Smooth 25fps video output
- Basic lip movement matching speech timing
- Stable WebRTC connection

❌ **May not be perfect**:
- Detailed phoneme matching (model limitation)
- Sub-300ms latency (requires optimization)
- Idle animation quality (using simple breathing effect)

## Next Steps

1. **If timing is the issue**: Focus on latency optimization
2. **If phonemes don't match**: This is a model quality issue - consider using different source images or model fine-tuning
3. **If video is choppy**: Optimize rendering pipeline or check network
4. **If everything works**: Proceed with production deployment optimizations

## Reference: Good Lip Sync Examples

To calibrate your expectations, compare against:
- Professional lip-dubbed movies (gold standard)
- Video conferencing apps (200-300ms is normal)
- Speech-driven animation in games (300-500ms common)
- Zoom/Google Meet video calls (check your own latency)

Remember: Perfect sync (<100ms) is very difficult in real-time systems. Your current ~550ms is in the acceptable range for interactive avatars.
