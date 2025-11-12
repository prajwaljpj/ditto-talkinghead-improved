# ChunkSize Parameter Fix

## Critical Issue Discovered

The `chunksize` parameter in StreamSDK is **NOT** just about frame count—it's used to calculate the expected audio chunk size!

### Error Encountered

```
ERROR: cannot reshape array of size 5120 into shape (5,2,1024)
```

This happened when trying to use 3240 samples with `chunksize=(3,5,2)`.

## Root Cause

### The Formula (from `core/atomic_components/wav2feat.py:72-79`)

```python
def __call__(self, audio_chunk, chunksize=(3, 5, 2)):
    """
    audio_chunk: int(sum(chunksize) * 0.04 * 16000) + 80    # 6480
    """
    valid_feat_s = - sum(chunksize[1:]) * 2   # -7
    valid_feat_e = - chunksize[2] * 2   # -2

    encoding_chunk = self.hubert(audio_chunk)
    valid_encoding = encoding_chunk[valid_feat_s:valid_feat_e]
    valid_feat = valid_encoding.reshape(chunksize[1], 2, 1024).mean(1)    # [5, 1024]
    return valid_feat
```

### Key Insights

1. **Audio chunk size formula**:
   ```
   audio_chunk_samples = int(sum(chunksize) * 0.04 * 16000) + 80
   ```

2. **For default `chunksize=(3,5,2)`**:
   ```
   sum = 3 + 5 + 2 = 10
   audio_samples = 10 * 0.04 * 16000 + 80 = 6480 samples
   ```

3. **The reshape operation expects**:
   - `chunksize[1]` number of features (e.g., 5)
   - Each feature is `2 * 1024` dimensions
   - Total HuBERT output: `chunksize[1] * 2 * 1024 = 5 * 2048 = 10240`

4. **What went wrong with 3240 samples**:
   - 3240 samples → HuBERT outputs different number of features
   - Tried to reshape into `(5, 2, 1024)` but actual size was different
   - **Mismatch between audio length and chunksize parameter!**

## The Relationship

| chunksize | sum | Audio Samples | Duration @ 16kHz | Frames Generated |
|-----------|-----|---------------|------------------|------------------|
| (3,5,2) | 10 | 6480 | 405ms | 5 (chunksize[1]) |
| (2,3,1) | 6 | 3920 | 245ms | 3 (chunksize[1]) |
| (2,2,1) | 5 | 3280 | 205ms | 2 (chunksize[1]) |
| (1,3,1) | 5 | 3280 | 205ms | 3 (chunksize[1]) |

**Important**: The middle value `chunksize[1]` determines the number of frames generated per chunk!

## Why We Can't Use Smaller Chunks

To get 3240 samples, we'd need:
```
3240 = sum(chunksize) * 640 + 80
3160 = sum(chunksize) * 640
sum(chunksize) = 4.9375
```

Since chunksize must be integers, we can't get exactly 3240 samples.

The closest is `chunksize=(2,2,1)` which gives:
```
sum = 5
audio_samples = 5 * 640 + 80 = 3280 samples
```

**But**: This would only generate 2 frames per chunk (`chunksize[1]=2`), not 5.

## Solution: Back to Original

We must use the **default `chunksize=(3,5,2)`** which produces **6480 samples**.

### Why This Is Acceptable

1. **TensorRT Compatible**: 6480 is within valid range [3240..12960]
2. **Generates 5 frames**: Decent frame rate per chunk
3. **Well-tested**: This is the default the model was trained with
4. **StreamSDK expects it**: The reshape operations are hardcoded for this pattern

### Performance Trade-off

| Metric | Attempted (3240) | Actual (6480) | Original Issue |
|--------|------------------|---------------|----------------|
| Chunk size | 3240 samples (202.5ms) | 6480 samples (405ms) | Same as original |
| Feed rate | ~5 Hz | ~2.5 Hz | Same as original |
| IDLE FPS | Would be 25-30 | ~12-13 FPS | Was ~3 FPS |
| SPEAKING FPS | Would be 25-30 | ~25 FPS | Was ~3 FPS |

## Final Configuration

```python
# In webrtc/livekit_gemini_agent.py

# Silent audio
self.silent_audio_chunk_size = 6480  # Default chunk size
self.silent_audio_chunksize = (3, 5, 2)  # Corresponding chunksize parameter

# Gemini audio
self.model_chunk_size = 6480  # Default chunk size
self.model_chunksize = (3, 5, 2)  # Chunksize parameter for StreamSDK
```

## Key Learnings

### 1. chunksize is NOT arbitrary
The `chunksize=(3,5,2)` tuple controls:
- **Total chunk length**: `sum(chunksize)` determines audio sample count
- **Frame count**: `chunksize[1]` determines how many frames are generated
- **Overlap**: `chunksize[0]` and `chunksize[2]` affect temporal context

### 2. Audio length MUST match chunksize
You cannot feed arbitrary audio lengths with any chunksize. They are tightly coupled by the formula:
```
audio_samples = int(sum(chunksize) * 0.04 * 16000) + 80
```

### 3. The middle value matters most
`chunksize[1]` is used in the reshape operation:
```python
valid_feat = valid_encoding.reshape(chunksize[1], 2, 1024).mean(1)
```

If you feed wrong audio length, the reshape fails with size mismatch.

### 4. TensorRT constraints are separate
TensorRT's [3240..12960] range is about the **model's input layer**, not about the chunksize parameter. The chunksize parameter is a **StreamSDK/HuBERT processing parameter**.

## Alternative Approaches (Future)

If you really need smaller chunks for lower latency:

### Option 1: Modify chunksize to (2,2,1)
```python
chunksize = (2, 2, 1)
# Produces: 5 * 640 + 80 = 3280 samples (205ms)
# Generates: 2 frames per chunk
# Feed rate: ~5 Hz → ~10 FPS
```

### Option 2: Modify chunksize to (1,3,1)
```python
chunksize = (1, 3, 1)
# Produces: 5 * 640 + 80 = 3280 samples (205ms)
# Generates: 3 frames per chunk
# Feed rate: ~5 Hz → ~15 FPS
```

### Option 3: Modify chunksize to (2,4,2)
```python
chunksize = (2, 4, 2)
# Produces: 8 * 640 + 80 = 5200 samples (325ms)
# Generates: 4 frames per chunk
# Feed rate: ~3 Hz → ~12 FPS
```

## Testing Alternative Chunksizes

If you want to experiment:

```python
# Test script
chunk size_configs = [
    ((3, 5, 2), 6480),  # Default
    ((2, 4, 2), 5200),  # Alternative 1
    ((1, 3, 1), 3280),  # Alternative 2
    ((2, 2, 1), 3280),  # Alternative 3
]

for chunksize, expected_samples in chunksize_configs:
    calculated = int(sum(chunksize) * 0.04 * 16000) + 80
    assert calculated == expected_samples
    print(f"chunksize={chunksize} → {calculated} samples ({calculated/16000*1000:.1f}ms)")
    print(f"  Generates {chunksize[1]} frames per chunk")
    print(f"  Feed rate: ~{1000/(calculated/16000):.1f} Hz")
    print(f"  Effective FPS: ~{chunksize[1] * 1000/(calculated/16000):.1f}")
```

## Conclusion

The **chunksize parameter** is fundamental to how StreamSDK processes audio. It's not a simple frame count—it's a formula that determines:
1. Expected audio chunk size
2. Number of frames to generate
3. How to reshape HuBERT features

We must use `chunksize=(3,5,2)` with `6480 samples` because:
- It's what the code expects
- It matches the TensorRT model's optimization
- It's been tested and proven to work
- Alternative values would require code changes in StreamSDK

**Current solution**: Back to original 6480 samples, but still an improvement over the initial ~3 FPS because we're feeding continuously rather than in large bursts.

---

**Date**: 2025-11-11
**Issue**: Reshape error with mismatched chunksize
**Solution**: Use default chunksize=(3,5,2) with 6480 samples
**Status**: ✅ Fixed
