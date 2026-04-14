"""
Post-Processing Module
- Feathered alpha blending at inpainted boundary
- Color/tone correction in LAB space
- Optional unsharp masking for texture restoration
"""

import logging
from typing import Optional
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class PostProcessor:
    """
    Applies post-processing steps to the inpainted image to:
    1. Seamlessly blend inpainted region with original using feathered alpha
    2. Correct any color drift introduced by inpainting
    3. Restore sharpness lost due to blending/inpainting
    """

    def process(
        self,
        image: np.ndarray,
        original: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        sharpen: bool = True,
        color_correct: bool = True
    ) -> np.ndarray:
        """
        Args:
            image:        Inpainted RGB image (uint8)
            original:     Original image before inpainting (for color reference)
            mask:         Inpainting mask (for targeted color correction)
            sharpen:      Apply unsharp masking
            color_correct: Apply color correction
        
        Returns:
            Post-processed RGB image (uint8)
        """
        result = image.copy()

        # Ensure shapes match before any operation
        if original is not None and mask is not None:
            if image.shape[:2] != original.shape[:2]:
                h, w = image.shape[:2]
                original = cv2.resize(original, (w, h), interpolation=cv2.INTER_AREA)
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            if image.shape[:2] != mask.shape[:2]:
                h, w = image.shape[:2]
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Step 1: Feathered alpha blend (inpainted inside mask, original outside)
        if original is not None and mask is not None:
            result = self._feathered_blend(result, original, mask)

        # Step 2: Color correction
        if color_correct and original is not None and mask is not None:
            result = self._color_correct(
                result.astype(np.float32),
                original.astype(np.float32),
                mask
            )
            result = np.clip(result, 0, 255).astype(np.uint8)

        # Step 3: Sharpen
        if sharpen:
            result = self._unsharp_mask(result.astype(np.float32))
            result = np.clip(result, 0, 255).astype(np.uint8)

        return result

    def _feathered_blend(
        self,
        inpainted: np.ndarray,
        original: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Blend inpainted and original using a feathered (Gaussian-blurred) mask.
        - Inside mask: 100% inpainted (watermark removed)
        - Outside mask: 100% original (untouched)
        - At boundary: smooth gradient transition (no visible seam)
        """
        # Create feathered alpha from the mask
        # Gaussian blur creates a smooth falloff at edges
        alpha = mask.astype(np.float32) / 255.0
        alpha = cv2.GaussianBlur(alpha, (21, 21), 5)
        
        # Expand to 3 channels
        alpha_3ch = alpha[:, :, np.newaxis]
        
        # Blend: inpainted where mask=1, original where mask=0
        result = (inpainted.astype(np.float32) * alpha_3ch +
                  original.astype(np.float32) * (1.0 - alpha_3ch))
        
        return np.clip(result, 0, 255).astype(np.uint8)

    def _color_correct(
        self,
        inpainted: np.ndarray,
        original: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Match inpainted region luminance to surroundings while preserving hue.
        """
        mask_bool = mask > 127
        if not mask_bool.any():
            return inpainted

        # Use LAB color space for better luminance control
        inp_lab = cv2.cvtColor(inpainted.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        ref_lab = cv2.cvtColor(original.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)

        # Stats from unmasked region for reference
        ref_l = ref_lab[~mask_bool][:, 0]
        inp_l = inp_lab[mask_bool][:, 0]

        if len(ref_l) == 0 or len(inp_l) == 0:
            return inpainted

        # Shift luminance to match room lighting
        l_shift = ref_l.mean() - inp_l.mean()
        l_shift = np.clip(l_shift, -30, 30)
        inp_lab[mask_bool, 0] = np.clip(inp_lab[mask_bool, 0] + l_shift, 0, 255)

        corrected_rgb = cv2.cvtColor(inp_lab.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)

        # Very gentle blend — 30% corrected, 70% inpainted
        result = inpainted.copy()
        alpha = 0.3
        result[mask_bool] = (1 - alpha) * inpainted[mask_bool] + alpha * corrected_rgb[mask_bool]

        return result

    def _unsharp_mask(
        self,
        img: np.ndarray,
        sigma: float = 1.0,
        strength: float = 0.3,
        threshold: int = 3
    ) -> np.ndarray:
        """
        Unsharp masking to restore fine texture detail.
        Gentle settings to avoid over-sharpening inpainted regions.
        """
        blurred = cv2.GaussianBlur(img, (0, 0), sigma)
        sharpened = img + strength * (img - blurred)

        if threshold > 0:
            low_contrast = np.abs(img - blurred).max(axis=2) < threshold
            sharpened[low_contrast] = img[low_contrast]

        return sharpened