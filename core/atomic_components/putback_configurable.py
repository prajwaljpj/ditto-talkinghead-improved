"""
Configurable PutBack with different blending modes.

Allows control over how much of the rendered face vs original face is used.
"""

import cv2
import numpy as np
from ..utils.blend import blend_images_cy
from ..utils.get_mask import get_mask


class PutBackConfigurable:
    """
    Configurable face compositing with different blending modes.

    Modes:
    - 'blend': Default smooth blending (feathered edges)
    - 'replace': Full replacement (no blending, hard edges)
    - 'strong': More rendered face, less original (90% rendered)
    - 'weak': More original face, less rendered (50% rendered)
    - 'custom': Custom alpha value
    """

    def __init__(
        self,
        mask_template_path=None,
        blend_mode='blend',
        custom_alpha=None,
        mask_ratio_w=0.9,
        mask_ratio_h=0.9,
    ):
        """
        Args:
            mask_template_path: Path to custom mask template (optional)
            blend_mode: 'blend', 'replace', 'strong', 'weak', or 'custom'
            custom_alpha: Alpha value for 'custom' mode (0.0-1.0)
            mask_ratio_w: Mask width ratio (0.0-1.0, default 0.9)
            mask_ratio_h: Mask height ratio (0.0-1.0, default 0.9)
        """
        self.blend_mode = blend_mode
        self.custom_alpha = custom_alpha

        # Generate or load mask
        if mask_template_path is None:
            mask = get_mask(512, 512, mask_ratio_w, mask_ratio_h)
            mask = np.concatenate([mask] * 3, 2)
        else:
            mask = cv2.imread(mask_template_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0

        # Apply blend mode to mask
        self.mask_ori_float = self._apply_blend_mode(mask[:,:,0])
        self.result_buffer = None

    def _apply_blend_mode(self, mask):
        """Apply blending mode to the mask."""
        if self.blend_mode == 'blend':
            # Default smooth blending
            return mask

        elif self.blend_mode == 'replace':
            # Full replacement: all 1s (100% rendered face)
            return np.ones_like(mask)

        elif self.blend_mode == 'strong':
            # Strong rendered face: increase mask values
            # Map [0, 1] → [0.5, 1.0]
            return mask * 0.5 + 0.5

        elif self.blend_mode == 'weak':
            # Weak rendered face: decrease mask values
            # Map [0, 1] → [0.0, 0.5]
            return mask * 0.5

        elif self.blend_mode == 'custom':
            # Custom alpha blending
            if self.custom_alpha is None:
                raise ValueError("custom_alpha must be provided for 'custom' blend mode")
            # Blend between original mask and full mask
            alpha = np.clip(self.custom_alpha, 0.0, 1.0)
            return mask * alpha + (1 - mask) * (1 - alpha)

        else:
            raise ValueError(f"Unknown blend_mode: {self.blend_mode}")

    def __call__(self, frame_rgb, render_image, M_c2o):
        """
        Composite rendered face onto original frame.

        Args:
            frame_rgb: Original frame (H, W, 3)
            render_image: Rendered face (512, 512, 3)
            M_c2o: Transformation matrix from crop to original

        Returns:
            result: Composited frame (H, W, 3)
        """
        h, w = frame_rgb.shape[:2]

        # Warp mask to original frame size
        mask_warped = cv2.warpAffine(
            self.mask_ori_float, M_c2o[:2, :], dsize=(w, h), flags=cv2.INTER_LINEAR
        ).clip(0, 1)

        # Warp rendered face to original frame size
        frame_warped = cv2.warpAffine(
            render_image, M_c2o[:2, :], dsize=(w, h), flags=cv2.INTER_LINEAR
        )

        # Allocate result buffer
        self.result_buffer = np.empty((h, w, 3), dtype=np.uint8)

        # Blend images
        blend_images_cy(mask_warped, frame_warped, frame_rgb, self.result_buffer)

        return self.result_buffer


def get_putback_by_mode(blend_mode='blend', custom_alpha=None, mask_ratio_w=0.9, mask_ratio_h=0.9):
    """
    Factory function to get PutBack with specified blend mode.

    Args:
        blend_mode: 'blend', 'replace', 'strong', 'weak', or 'custom'
        custom_alpha: Alpha value for 'custom' mode (0.0-1.0)
        mask_ratio_w: Mask width ratio
        mask_ratio_h: Mask height ratio

    Returns:
        PutBackConfigurable instance
    """
    return PutBackConfigurable(
        blend_mode=blend_mode,
        custom_alpha=custom_alpha,
        mask_ratio_w=mask_ratio_w,
        mask_ratio_h=mask_ratio_h,
    )
