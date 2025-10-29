# Face Blending Control Guide

## Overview

This guide explains how to control the blending between the **rendered face** (with lip-sync) and the **original face** from your source video.

By default, Ditto blends the rendered face with the original face using a smooth gradient. However, you can control exactly how much of each face is used in the final output.

---

## Table of Contents

1. [Understanding Face Compositing](#understanding-face-compositing)
2. [Blend Modes](#blend-modes)
3. [Usage Examples](#usage-examples)
4. [Visual Comparison](#visual-comparison)
5. [When to Use Each Mode](#when-to-use-each-mode)
6. [Common Scenarios](#common-scenarios)
7. [Technical Details](#technical-details)

---

## Understanding Face Compositing

### What Gets Blended?

```
┌─────────────────────────────────────────────────┐
│ Original Frame (from your video)                │
│  ┌──────────────┐                              │
│  │   Your Face  │                              │
│  │   • Eyes     │ ← Mostly preserved           │
│  │   • Nose     │ ← Mostly preserved           │
│  │   • Mouth 😊 │ ← BLENDED with rendered     │
│  │   • Skin     │ ← Preserved                  │
│  └──────────────┘                              │
│  Background/Body ← Completely preserved        │
└─────────────────────────────────────────────────┘
                    +
┌─────────────────────────────────────────────────┐
│ Rendered Face (generated with lip-sync)         │
│  ┌──────────────┐                              │
│  │   Same Face  │                              │
│  │   • Eyes     │ ← Can be used               │
│  │   • Nose     │ ← Can be used               │
│  │   • Mouth 🗣️ │ ← NEW (talking)             │
│  │   • Skin     │ ← Generated                  │
│  └──────────────┘                              │
└─────────────────────────────────────────────────┘
                    =
┌─────────────────────────────────────────────────┐
│ Output Frame (blended result)                   │
│  ┌──────────────┐                              │
│  │   Blended    │                              │
│  │   Face       │                              │
│  │   (talking)  │ ← Blend ratio determined    │
│  │              │   by blend mode              │
│  └──────────────┘                              │
│  Original Background/Body                       │
└─────────────────────────────────────────────────┘
```

### The Blending Formula

```python
result = mask * rendered_face + (1 - mask) * original_face
```

Where:
- `mask` = blending weight (0.0 to 1.0)
- `mask = 1.0` → 100% rendered face
- `mask = 0.0` → 100% original face
- `mask = 0.5` → 50/50 blend

---

## Blend Modes

### Available Modes

| Mode | Center Blend | Edge Blend | Description |
|------|--------------|------------|-------------|
| **blend** | 50% rendered | Smooth gradient | Default, natural looking |
| **strong** | 90% rendered | Moderate gradient | Strong lip-sync effect |
| **replace** | 100% rendered | No gradient | Full replacement |
| **weak** | 50% rendered | Soft gradient | Subtle effect |
| **custom** | User-defined | Adjustable | Precise control |

### Mode Details

#### 1. **blend** (Default)
```
Blending Profile:
┌────────────────────────────────┐
│        Face Region             │
│  Edges:  0% → 25% → 50%       │
│  Center: 50% rendered          │
│  Edges:  50% → 25% → 0%       │
└────────────────────────────────┘

Characteristics:
• Natural appearance
• Smooth transitions at edges
• Balanced between original and rendered
• Best for most use cases

Use when:
✓ You want natural results
✓ First time trying Ditto
✓ Original face is good quality
```

#### 2. **strong**
```
Blending Profile:
┌────────────────────────────────┐
│        Face Region             │
│  Edges:  50% → 75% → 90%      │
│  Center: 90% rendered          │
│  Edges:  90% → 75% → 50%      │
└────────────────────────────────┘

Characteristics:
• Strong lip-sync visibility
• More of rendered face used
• Less of original face artifacts
• Slightly harder edges

Use when:
✓ Original has distracting elements (smile, teeth)
✓ You want prominent lip-sync
✓ Default blend is too subtle
```

#### 3. **replace**
```
Blending Profile:
┌────────────────────────────────┐
│        Face Region             │
│  All:    100% rendered         │
│  No blending/gradient          │
└────────────────────────────────┘

Characteristics:
• Complete replacement
• Maximum fidelity to generated motion
• Hard edges at face boundary
• No original face visible

Use when:
✓ You want exact generated motion
✓ Don't care about edge smoothness
✓ Original face has major issues
⚠️  May look unnatural at edges
```

#### 4. **weak**
```
Blending Profile:
┌────────────────────────────────┐
│        Face Region             │
│  Edges:  0% → 10% → 25%       │
│  Center: 25-50% rendered       │
│  Edges:  25% → 10% → 0%       │
└────────────────────────────────┘

Characteristics:
• Subtle lip-sync effect
• More original face preserved
• Very soft blending
• Maintains original character

Use when:
✓ Original face is high quality
✓ You want subtle enhancement
✓ Preserving original is priority
```

#### 5. **custom**
```
Blending Profile:
┌────────────────────────────────┐
│        Face Region             │
│  Controlled by --blend_alpha   │
│  Alpha range: 0.0 - 1.0        │
└────────────────────────────────┘

Characteristics:
• Precise percentage control
• Adjustable with alpha parameter
• Fine-tuning capability
• Experimental freedom

Alpha Values:
• 0.0 = 100% original (no effect)
• 0.3 = 30% rendered (very subtle)
• 0.5 = 50/50 blend (balanced)
• 0.7 = 70% rendered (strong)
• 1.0 = 100% rendered (full replace)
```

---

## Usage Examples

### Script: `inference_video_blend_control.py`

#### Example 1: Default Blending
```bash
python3 inference_video_blend_control.py \
    --source_video avatar.mp4 \
    --audio speech.wav \
    --output result_blend.mp4
```

#### Example 2: Strong Blending (90% rendered)
```bash
python3 inference_video_blend_control.py \
    --source_video avatar.mp4 \
    --audio speech.wav \
    --output result_strong.mp4 \
    --blend_mode strong
```

#### Example 3: Full Replacement (100% rendered)
```bash
python3 inference_video_blend_control.py \
    --source_video avatar.mp4 \
    --audio speech.wav \
    --output result_replace.mp4 \
    --blend_mode replace
```

#### Example 4: Custom 80% Rendered
```bash
python3 inference_video_blend_control.py \
    --source_video avatar.mp4 \
    --audio speech.wav \
    --output result_80.mp4 \
    --blend_mode custom \
    --blend_alpha 0.8
```

#### Example 5: Weak Blending (Subtle)
```bash
python3 inference_video_blend_control.py \
    --source_video avatar.mp4 \
    --audio speech.wav \
    --output result_weak.mp4 \
    --blend_mode weak
```

#### Example 6: Combined with Template Frames
```bash
# Use first 100 frames for motion + strong blending
python3 inference_video_blend_control.py \
    --source_video looping_avatar.mp4 \
    --audio speech.wav \
    --output result.mp4 \
    --template_n_frames 100 \
    --blend_mode strong
```

---

## Visual Comparison

### Side-by-Side Comparison

```
┌──────────────────────────────────────────────────────────────┐
│                    ORIGINAL FRAME                            │
│  Face with smile 😊, teeth showing, neutral expression       │
└──────────────────────────────────────────────────────────────┘
                            ↓
        ┌───────────────────┴───────────────────┐
        │                                       │
┌───────▼────────┐  ┌──────────────┐  ┌───────▼────────┐
│ blend (50%)    │  │ strong (90%) │  │ replace (100%) │
│                │  │              │  │                │
│ • Smooth       │  │ • Stronger   │  │ • Complete     │
│ • Natural      │  │ • Less smile │  │ • No smile     │
│ • Some smile   │  │ • Clear lips │  │ • Hard edges   │
│ • Soft edges   │  │ • Good edges │  │ • Max fidelity │
└────────────────┘  └──────────────┘  └────────────────┘

┌───────────────┐  ┌────────────────┐
│ weak (25%)    │  │ custom (70%)   │
│               │  │                │
│ • Very subtle │  │ • Adjustable   │
│ • Most smile  │  │ • Balanced     │
│ • Soft effect │  │ • Flexible     │
└───────────────┘  └────────────────┘
```

### Mask Visualization

```
BLEND MODE: 'blend'
┌────────────────────────┐
│ ░░░░░░░░░░░░░░░░░░░░░ │ 0% (original)
│ ░░░████████████░░░     │ 25%
│ ████████████████████   │ 50%
│ ████████████████████   │ 50%
│ ░░░████████████░░░     │ 25%
│ ░░░░░░░░░░░░░░░░░░░░░ │ 0%
└────────────────────────┘

BLEND MODE: 'strong'
┌────────────────────────┐
│ ░░░████████████░░░     │ 50%
│ ████████████████████   │ 75%
│ ████████████████████   │ 90%
│ ████████████████████   │ 90%
│ ████████████████████   │ 75%
│ ░░░████████████░░░     │ 50%
└────────────────────────┘

BLEND MODE: 'replace'
┌────────────────────────┐
│ ████████████████████   │ 100%
│ ████████████████████   │ 100%
│ ████████████████████   │ 100%
│ ████████████████████   │ 100%
│ ████████████████████   │ 100%
│ ████████████████████   │ 100%
└────────────────────────┘

Legend:
░ = Original face
█ = Rendered face
```

---

## When to Use Each Mode

### Decision Tree

```
Do you want natural looking results?
    ├─ Yes → Use 'blend' (default)
    └─ No
        │
        Does original face have problems (smile, teeth)?
        ├─ Yes → Use 'strong' or 'replace'
        └─ No
            │
            Do you want subtle lip-sync?
            ├─ Yes → Use 'weak'
            └─ No → Use 'custom' to fine-tune
```

### Use Case Matrix

| Original Face | Desired Effect | Recommended Mode |
|---------------|----------------|------------------|
| Good quality, neutral | Natural lip-sync | **blend** |
| Has smile/teeth | Hide smile, strong lips | **strong** |
| Low quality | Replace completely | **replace** |
| High quality | Enhance subtly | **weak** |
| Any | Experiment/fine-tune | **custom** |

### Detailed Recommendations

#### Use **'blend'** when:
✅ Original face is good quality
✅ Neutral or mild expression
✅ You want natural results
✅ First time using the tool
✅ Not sure which mode to use

#### Use **'strong'** when:
✅ Original has smile/teeth showing
✅ You want prominent lip-sync
✅ Default blend too subtle
✅ Original has artifacts
✅ Lip movements should dominate

#### Use **'replace'** when:
✅ Maximum fidelity to generated motion
✅ Original face has major issues
✅ You don't care about edge quality
✅ Testing/debugging generated motion
⚠️  Accept potential hard edges

#### Use **'weak'** when:
✅ Original face is perfect
✅ Want minimal modification
✅ Preserve original character
✅ Subtle enhancement only
✅ Original expression good

#### Use **'custom'** when:
✅ None of the presets work
✅ You want precise control
✅ Experimenting with ratios
✅ Different parts of video need different blends
✅ Fine-tuning for specific use case

---

## Common Scenarios

### Scenario 1: Avatar with Smile

**Problem**: Source video has smile with visible teeth. Generated lip-sync looks bad.

**Solution**:
```bash
python3 inference_video_blend_control.py \
    --source_video smiling_avatar.mp4 \
    --audio speech.wav \
    --output result.mp4 \
    --template_n_frames 100 \
    --blend_mode strong
```

**Why**:
- `template_n_frames 100`: Uses only neutral frames for motion
- `blend_mode strong`: Reduces smile visibility (90% rendered)

### Scenario 2: High-Quality Portrait

**Problem**: Original face is perfect. Want to preserve it but add lip-sync.

**Solution**:
```bash
python3 inference_video_blend_control.py \
    --source_video portrait.jpg \
    --audio speech.wav \
    --output result.mp4 \
    --blend_mode weak
```

**Why**:
- `blend_mode weak`: Preserves most of original, adds subtle lip-sync

### Scenario 3: Low-Quality Video

**Problem**: Original video has artifacts, noise, or poor quality.

**Solution**:
```bash
python3 inference_video_blend_control.py \
    --source_video low_quality.mp4 \
    --audio speech.wav \
    --output result.mp4 \
    --blend_mode replace
```

**Why**:
- `blend_mode replace`: Uses 100% rendered face, ignores poor original

### Scenario 4: Looping Avatar Video

**Problem**: Avatar video loops through neutral → smile → neutral.

**Solution 1: Use neutral frames only**
```bash
python3 inference_video_blend_control.py \
    --source_video looping_avatar.mp4 \
    --audio speech.wav \
    --output result.mp4 \
    --template_n_frames 100
```

**Solution 2: Use all frames with strong blend**
```bash
python3 inference_video_preserve_motion.py \
    --source_video looping_avatar.mp4 \
    --audio speech.wav \
    --output result.mp4 \
    --motion_frames 100
```

Then apply blending:
```bash
python3 inference_video_blend_control.py \
    --source_video looping_avatar.mp4 \
    --audio speech.wav \
    --output result_strong.mp4 \
    --blend_mode strong
```

### Scenario 5: Testing Different Blends

**Problem**: Not sure which blend looks best.

**Solution**: Generate multiple versions
```bash
#!/bin/bash
MODES=("blend" "strong" "replace" "weak")

for mode in "${MODES[@]}"; do
    python3 inference_video_blend_control.py \
        --source_video avatar.mp4 \
        --audio speech.wav \
        --output "result_${mode}.mp4" \
        --blend_mode "$mode"
done

# Custom alphas
for alpha in 0.3 0.5 0.7 0.9; do
    python3 inference_video_blend_control.py \
        --source_video avatar.mp4 \
        --audio speech.wav \
        --output "result_alpha_${alpha}.mp4" \
        --blend_mode custom \
        --blend_alpha "$alpha"
done
```

---

## Technical Details

### The Compositing Process

#### Step 1: Create Mask
```python
# Default mask (feathered gradient)
mask = get_mask(512, 512, ratio_w=0.9, ratio_h=0.9)
```

#### Step 2: Apply Blend Mode
```python
if blend_mode == 'blend':
    # Use mask as-is
    final_mask = mask

elif blend_mode == 'strong':
    # Increase mask values: [0,1] → [0.5,1]
    final_mask = mask * 0.5 + 0.5

elif blend_mode == 'replace':
    # All ones (full replacement)
    final_mask = np.ones_like(mask)

elif blend_mode == 'weak':
    # Decrease mask values: [0,1] → [0,0.5]
    final_mask = mask * 0.5

elif blend_mode == 'custom':
    # Custom alpha blending
    final_mask = mask * alpha + (1 - mask) * (1 - alpha)
```

#### Step 3: Transform to Original Frame
```python
# Warp mask to match original frame size and position
mask_warped = cv2.warpAffine(
    final_mask, M_c2o[:2, :],
    dsize=(width, height),
    flags=cv2.INTER_LINEAR
)
```

#### Step 4: Warp Rendered Face
```python
# Warp rendered face to original frame position
frame_warped = cv2.warpAffine(
    render_image, M_c2o[:2, :],
    dsize=(width, height),
    flags=cv2.INTER_LINEAR
)
```

#### Step 5: Blend
```python
# Alpha blending
result = mask_warped * frame_warped + (1 - mask_warped) * original_frame
```

### Mask Generation Parameters

```python
def get_mask(W, H, ratio_w=0.9, ratio_h=0.9):
    """
    Generate feathered mask for face region.

    Args:
        W: Width (512)
        H: Height (512)
        ratio_w: Width ratio (0.9 = 90% of width)
        ratio_h: Height ratio (0.9 = 90% of height)

    Returns:
        mask: Gradient mask (H, W, 1) with values 0.0-1.0
              1.0 = full replacement
              0.0 = full original
    """
```

The mask has:
- **Center region**: High values (close to 1.0)
- **Edge regions**: Gradient (0.0 to 1.0)
- **Corners**: Low values (close to 0.0)

This creates smooth blending at face boundaries.

---

## Custom Alpha Guide

For `--blend_mode custom`, use `--blend_alpha` to specify the blend ratio:

```
Alpha Value | Effect                     | Use Case
------------|----------------------------|---------------------------
0.0         | 100% original (no change) | Testing/debugging
0.1-0.2     | Very subtle lip-sync       | High-quality original
0.3-0.4     | Subtle enhancement         | Good original, light touch
0.5         | Balanced 50/50             | Equal weight
0.6-0.7     | Noticeable lip-sync        | Default alternative
0.8-0.9     | Strong lip-sync            | Problematic original
1.0         | Full replacement           | Same as 'replace' mode
```

### Finding Your Sweet Spot

Try these alphas in order:
1. Start with `0.5` (balanced)
2. If too subtle: increase to `0.7`
3. If still subtle: try `0.9`
4. If too strong: decrease to `0.3`
5. Fine-tune by `±0.1` increments

Example workflow:
```bash
# Start balanced
python3 inference_video_blend_control.py ... --blend_mode custom --blend_alpha 0.5

# Too subtle? Try stronger
python3 inference_video_blend_control.py ... --blend_mode custom --blend_alpha 0.7

# Still subtle? Go stronger
python3 inference_video_blend_control.py ... --blend_mode custom --blend_alpha 0.9

# Too strong? Go back
python3 inference_video_blend_control.py ... --blend_mode custom --blend_alpha 0.6
```

---

## Summary

### Quick Reference

| Goal | Command |
|------|---------|
| Natural results | `--blend_mode blend` (default) |
| Hide smile/teeth | `--blend_mode strong` |
| Maximum lip-sync | `--blend_mode replace` |
| Preserve original | `--blend_mode weak` |
| Fine-tune | `--blend_mode custom --blend_alpha 0.7` |

### Best Practices

1. **Start with defaults**: Try `blend` mode first
2. **Iterate**: Generate multiple versions to compare
3. **Match to source**: Higher quality original = lower blend ratio
4. **Consider edges**: `replace` mode may have hard edges
5. **Test short clips**: Try 5-10 seconds before full video

### Common Mistakes

❌ Using `replace` mode without checking edge quality
❌ Not testing multiple modes to compare
❌ Using same mode for all videos (each is different!)
❌ Ignoring original video quality when choosing mode
❌ Not combining with `template_n_frames` for problematic sources

✅ Test multiple modes on short clip first
✅ Choose mode based on original quality
✅ Use `strong` or `custom` for problematic originals
✅ Combine with frame selection for best results
✅ Fine-tune with custom alpha if needed

---

## Related Documentation

- [VIDEO_INFERENCE_GUIDE.md](./VIDEO_INFERENCE_GUIDE.md) - Basic video inference
- [INFERENCE_DOCUMENTATION.md](./INFERENCE_DOCUMENTATION.md) - Complete technical reference
- [TROUBLESHOOTING_SMILING_AVATAR.md](./TROUBLESHOOTING_SMILING_AVATAR.md) - Fix smiling avatars

---

**TL;DR**: Control how much rendered vs original face is used:
- `blend`: Natural (50/50)
- `strong`: More rendered (90/10)
- `replace`: All rendered (100/0)
- `weak`: Less rendered (25/75)
- `custom`: Your choice (use `--blend_alpha`)
