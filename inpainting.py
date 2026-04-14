"""
Inpainting Engine
Priority: LaMa → Simple-Lama-Inpainting → OpenCV NS (fallback)

LaMa (Large Mask inpainting) is ideal for real estate images:
- Handles large masks with global context
- No blur / patch artifacts
- Trained on diverse textures (walls, floors, sky, buildings)
"""

import logging
from typing import Optional

import numpy as np
import cv2

logger = logging.getLogger(__name__)


class InpaintingEngine:
    """
    Wraps multiple inpainting backends with automatic fallback:
      1. simple-lama-inpainting (pip-installable LaMa wrapper)
      2. lama-cleaner HTTP API (if running locally)
      3. OpenCV Navier-Stokes inpainting (last resort)
    """

    def __init__(self):
        self.model_name = "none"
        self._backend = None
        self._init_backend()

    def _init_backend(self):
        # Try simple_lama (lightweight pip package wrapping LaMa weights)
        if self._try_simple_lama():
            return
        # Try lama-cleaner REST API (if user is running it separately)
        if self._try_lama_cleaner_api():
            return
        # Try iopaint (successor to lama-cleaner)
        if self._try_iopaint():
            return
        logger.warning("No LaMa backend available. Falling back to OpenCV inpainting.")
        self.model_name = "opencv_ns"
        self._backend = "opencv"

    def _try_simple_lama(self) -> bool:
        try:
            from simple_lama_inpainting import SimpleLama
            self._lama = SimpleLama()
            self.model_name = "lama"
            self._backend = "simple_lama"
            logger.info("simple-lama-inpainting loaded successfully.")
            return True
        except ImportError:
            logger.info("simple-lama-inpainting not installed.")
            return False
        except Exception as e:
            logger.warning(f"simple-lama failed to load: {e}")
            return False

    def _try_lama_cleaner_api(self) -> bool:
        try:
            import requests
            r = requests.get("http://localhost:8080/health", timeout=2)
            if r.status_code == 200:
                self._backend = "lama_cleaner_api"
                self.model_name = "lama_cleaner"
                logger.info("lama-cleaner API detected at localhost:8080.")
                return True
        except Exception:
            pass
        return False

    def _try_iopaint(self) -> bool:
        try:
            import requests
            r = requests.get("http://localhost:8080/", timeout=2)
            if r.status_code == 200:
                self._backend = "iopaint_api"
                self.model_name = "iopaint_lama"
                logger.info("iopaint API detected.")
                return True
        except Exception:
            pass
        return False

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Remove watermark by inpainting masked region.

        Args:
            image: RGB uint8 numpy array
            mask:  uint8 numpy array (255 = watermark region to fill)

        Returns:
            RGB uint8 numpy array with watermark removed
        """
        if mask.max() == 0:
            logger.info("Empty mask — returning image unchanged.")
            return image

        if self._backend == "simple_lama":
            return self._inpaint_simple_lama(image, mask)
        elif self._backend == "lama_cleaner_api":
            return self._inpaint_lama_cleaner_api(image, mask)
        elif self._backend == "iopaint_api":
            return self._inpaint_iopaint_api(image, mask)
        else:
            logger.info("Using OpenCV fallback (quality will be lower).")
            return self._inpaint_opencv(image, mask)

    def _inpaint_simple_lama(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        from PIL import Image
        pil_img = Image.fromarray(image)
        pil_mask = Image.fromarray(mask)
        result = self._lama(pil_img, pil_mask)
        
        # simple-lama-inpainting pads the image to a multiple of 8 internally.
        # We need to crop it back to the original boundaries.
        result_np = np.array(result)
        h, w = image.shape[:2]
        if result_np.shape[:2] != (h, w):
            result_np = result_np[:h, :w]
            
        return result_np

    def _inpaint_lama_cleaner_api(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        import requests
        import io
        from PIL import Image

        img_buf = io.BytesIO()
        Image.fromarray(image).save(img_buf, format="PNG")
        mask_buf = io.BytesIO()
        Image.fromarray(mask).save(mask_buf, format="PNG")

        response = requests.post(
            "http://localhost:8080/inpaint",
            files={
                "image": ("image.png", img_buf.getvalue(), "image/png"),
                "mask": ("mask.png", mask_buf.getvalue(), "image/png"),
            },
            data={"model": "lama", "sampler": "plms"},
            timeout=120
        )
        if response.status_code == 200:
            result = Image.open(io.BytesIO(response.content)).convert("RGB")
            return np.array(result)
        else:
            logger.warning(f"lama-cleaner API error {response.status_code}. Falling back to OpenCV.")
            return self._inpaint_opencv(image, mask)

    def _inpaint_iopaint_api(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """iopaint (successor to lama-cleaner) REST API."""
        import requests
        import io
        from PIL import Image

        img_buf = io.BytesIO()
        Image.fromarray(image).save(img_buf, format="PNG")
        mask_buf = io.BytesIO()
        Image.fromarray(mask).save(mask_buf, format="PNG")

        response = requests.post(
            "http://localhost:8080/api/v1/inpaint",
            json={
                "image": img_buf.getvalue().hex(),
                "mask": mask_buf.getvalue().hex(),
                "model_name": "lama"
            },
            timeout=120
        )
        if response.status_code == 200:
            result = Image.open(io.BytesIO(response.content)).convert("RGB")
            return np.array(result)
        else:
            logger.warning("iopaint API failed. Falling back to OpenCV.")
            return self._inpaint_opencv(image, mask)

    def _inpaint_opencv(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        OpenCV Navier-Stokes inpainting.
        Reduced radius to prevent 'blurry blob' effect.
        """
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Multi-scale inpainting with very conservative settings
        h, w = bgr.shape[:2]
        small_bgr = cv2.resize(bgr, (w // 2, h // 2))
        small_mask = cv2.resize(mask, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
        
        # Use small radius (3 instead of 7) to keep textures sharp
        small_result = cv2.inpaint(small_bgr, small_mask, 2, cv2.INPAINT_NS)

        hint = cv2.resize(small_result, (w, h), interpolation=cv2.INTER_LANCZOS4)
        mask_3ch = cv2.merge([mask, mask, mask])
        blended_bgr = np.where(mask_3ch > 0, hint, bgr)

        # Telea inpainting with minimal radius (2) for detail preservation
        result_bgr = cv2.inpaint(blended_bgr.astype(np.uint8), mask, 2, cv2.INPAINT_TELEA)
        return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)