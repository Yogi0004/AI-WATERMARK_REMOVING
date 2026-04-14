"""
Watermark Detection Module — Robust Multi-Pass
- Multi-pass OCR (full image + center crop + contrast-enhanced)
- Semi-transparent overlay detection via color channel analysis
- Logo detection using contour + circle Hough analysis
- Aggressive but precise mask dilation
"""

import logging
from typing import Optional, Tuple, List

import numpy as np
import cv2

logger = logging.getLogger(__name__)

# Lazy imports to avoid hard crash if optional deps missing
_easyocr_reader = None
_tesseract_available = False


def _get_easyocr():
    global _easyocr_reader
    if _easyocr_reader is None:
        try:
            import easyocr
            _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            logger.info("EasyOCR loaded successfully.")
        except ImportError:
            logger.warning("EasyOCR not installed. Will try Tesseract.")
    return _easyocr_reader


def _check_tesseract():
    global _tesseract_available
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        _tesseract_available = True
        logger.info("Tesseract available as fallback OCR.")
    except Exception:
        logger.warning("Tesseract not available.")
    return _tesseract_available


class WatermarkDetector:
    """
    Detects semi-transparent text watermarks in property images.
    Strategy — multiple passes to catch every pixel:
      1. EasyOCR on full image
      2. EasyOCR on center 70% crop (higher zoom = better OCR for small text)
      3. EasyOCR on contrast-enhanced center crop
      4. Tesseract fallback on each
      5. Semi-transparent overlay detection (catches faint halo around text)
      6. Logo detection near detected text regions
      7. Aggressive morphological bridging
    """

    WATERMARK_KEYWORDS = [
        "magicbricks", "magic", "bricks", "99acres", "housing", "makaan",
        "nobroker", "no broker", "broker", "commonfloor", "squareyards",
        "proptiger", "sulekha", "olx", "quikr", "jll", "colliers", "cushman",
        "watermark", "property", "realty", "realestate", "homes", "estate",
        "preview", "sample", "demo", "stock", "shutterstock", "getty",
        "alamy", "dreamstime", "istock", "adobe", "fotolia", "123rf",
    ]

    # Brands that always have a logo/icon to the LEFT of the text
    LOGO_LEFT_BRANDS = [
        "nobroker", "broker", "magic", "magicbricks", "99acres",
        "housing", "squareyards", "proptiger", "commonfloor",
    ]

    def __init__(self, dilation_kernel: int = 5, center_fraction: float = 0.7):
        self.dilation_kernel = dilation_kernel
        self.center_fraction = center_fraction
        self.ocr_backend = self._init_ocr()

    def _init_ocr(self) -> str:
        if _get_easyocr() is not None:
            return "easyocr"
        if _check_tesseract():
            return "tesseract"
        return "none"

    # ── Main entry point ─────────────────────────────────────────────
    def detect_watermark(self, img: np.ndarray) -> np.ndarray:
        """
        Returns binary mask (uint8, 0/255) of watermark region.
        Uses multiple passes over different crops and preprocessed versions
        to maximize detection rate.
        """
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # ── Pass 1: Full image OCR ──
        detected_full = self._run_ocr_pass(img, mask, 0, 0, "full-image")

        # ── Pass 2: Center 70% crop ──
        cx0, cy0, cx1, cy1 = self._center_crop_coords(h, w, fraction=0.7)
        center_crop = img[cy0:cy1, cx0:cx1]
        detected_center = self._run_ocr_pass(center_crop, mask, cx0, cy0, "center-70%")

        # ── Pass 3: Contrast-enhanced center crop ──
        enhanced = self._enhance_for_ocr(center_crop)
        detected_enhanced = self._run_ocr_pass(enhanced, mask, cx0, cy0, "enhanced-center")

        # ── Pass 4: Wider 85% crop (catches off-center watermarks) ──
        wx0, wy0, wx1, wy1 = self._center_crop_coords(h, w, fraction=0.85)
        wide_crop = img[wy0:wy1, wx0:wx1]
        detected_wide = self._run_ocr_pass(wide_crop, mask, wx0, wy0, "wide-85%")

        detected = detected_full or detected_center or detected_enhanced or detected_wide

        if not detected:
            logger.info("All OCR passes failed. Falling back to center mask.")
            mask = self.generate_center_mask(img)
            return mask

        # ── Supplement: semi-transparent halo detection ──
        halo_mask = self._detect_semitransparent_overlay(img, mask)
        mask = cv2.bitwise_or(mask, halo_mask)

        # ── Supplement: circular logo detection near text ──
        logo_mask = self._detect_nearby_logos(img, mask)
        mask = cv2.bitwise_or(mask, logo_mask)

        # ── Morphological closing + dilation ──
        mask = self._dilate_mask(mask)

        # ── Final proximity bloom ──
        mask = self._proximity_bloom(img, mask)

        logger.info(f"Final mask coverage: {mask.mean():.4f}")
        return mask

    # ── OCR pass runner ──────────────────────────────────────────────
    def _run_ocr_pass(
        self,
        crop: np.ndarray,
        mask: np.ndarray,
        offset_x: int,
        offset_y: int,
        pass_name: str = ""
    ) -> bool:
        """Run EasyOCR then Tesseract on the given crop, accumulating into mask."""
        detected = False
        if self.ocr_backend == "easyocr":
            detected = self._detect_easyocr(crop, mask, offset_x, offset_y, pass_name)
        if not detected and self.ocr_backend in ("easyocr", "tesseract"):
            detected = self._detect_tesseract(crop, mask, offset_x, offset_y, pass_name)
        return detected

    def _enhance_for_ocr(self, img: np.ndarray) -> np.ndarray:
        """
        Create a contrast-enhanced version that makes semi-transparent
        watermarks much more visible to OCR.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # CLAHE (contrast-limited adaptive histogram equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)

        # Also try inverting — some watermarks are lighter than background
        inverted = cv2.bitwise_not(enhanced_gray)

        # Merge into 3-channel for EasyOCR
        enhanced = cv2.merge([enhanced_gray, enhanced_gray, enhanced_gray])
        return enhanced

    # ── EasyOCR ──────────────────────────────────────────────────────
    def _detect_easyocr(
        self,
        crop: np.ndarray,
        mask: np.ndarray,
        offset_x: int,
        offset_y: int,
        pass_name: str = ""
    ) -> bool:
        reader = _get_easyocr()
        if reader is None:
            return False
        try:
            results = reader.readtext(crop, detail=1, paragraph=False)
            found = False
            for (bbox, text, conf) in results:
                if conf < 0.05:
                    continue
                text_lower = text.lower().strip()
                is_watermark = (
                    any(kw in text_lower for kw in self.WATERMARK_KEYWORDS)
                    or conf > 0.3  # Any confident text in center is suspect
                )
                if is_watermark or len(text_lower) > 3:
                    pts = np.array(bbox, dtype=np.int32)

                    # Dynamic expansion based on text height
                    text_h = max(abs(pts[3, 1] - pts[0, 1]), abs(pts[2, 1] - pts[1, 1]), 10)
                    left_expand = max(20, int(text_h * 0.5))
                    right_expand = max(20, int(text_h * 0.5))
                    top_expand = max(5, int(text_h * 0.3))
                    bottom_expand = max(5, int(text_h * 0.3))

                    # Brand logos to the left → expand left much more
                    if any(kw in text_lower for kw in self.LOGO_LEFT_BRANDS):
                        left_expand = int(text_h * 3.5)

                    # Expand bounding box
                    pts[0, 0] = max(0, pts[0, 0] - left_expand)
                    pts[3, 0] = max(0, pts[3, 0] - left_expand)
                    pts[1, 0] = min(crop.shape[1], pts[1, 0] + right_expand)
                    pts[2, 0] = min(crop.shape[1], pts[2, 0] + right_expand)
                    pts[0, 1] = max(0, pts[0, 1] - top_expand)
                    pts[1, 1] = max(0, pts[1, 1] - top_expand)
                    pts[2, 1] = min(crop.shape[0], pts[2, 1] + bottom_expand)
                    pts[3, 1] = min(crop.shape[0], pts[3, 1] + bottom_expand)

                    pts[:, 0] += offset_x
                    pts[:, 1] += offset_y
                    cv2.fillPoly(mask, [pts], 255)
                    found = True
                    logger.info(f"[{pass_name}] EasyOCR: '{text}' conf={conf:.2f}")
            return found
        except Exception as e:
            logger.warning(f"[{pass_name}] EasyOCR error: {e}")
            return False

    # ── Tesseract ────────────────────────────────────────────────────
    def _detect_tesseract(
        self,
        crop: np.ndarray,
        mask: np.ndarray,
        offset_x: int,
        offset_y: int,
        pass_name: str = ""
    ) -> bool:
        try:
            import pytesseract
            from PIL import Image

            pil_crop = Image.fromarray(crop)
            data = pytesseract.image_to_data(pil_crop, output_type=pytesseract.Output.DICT)
            found = False
            n = len(data["text"])
            for i in range(n):
                text = data["text"][i].strip()
                conf = int(data["conf"][i])
                if not text or conf < 30:
                    continue
                text_lower = text.lower()
                is_watermark = any(kw in text_lower for kw in self.WATERMARK_KEYWORDS)
                if is_watermark or (len(text) > 2 and conf > 50):
                    bh = data["height"][i]

                    left_expand = max(20, int(bh * 0.5))
                    right_expand = max(20, int(bh * 0.5))
                    if any(kw in text_lower for kw in self.LOGO_LEFT_BRANDS):
                        left_expand = int(bh * 3.5)

                    x = max(0, data["left"][i] - left_expand) + offset_x
                    y = max(0, data["top"][i] - 5) + offset_y
                    bw = data["width"][i] + left_expand + right_expand
                    bh_expanded = bh + 10
                    cv2.rectangle(mask, (x, y), (x + bw, y + bh_expanded), 255, -1)
                    found = True
                    logger.info(f"[{pass_name}] Tesseract: '{text}' conf={conf}")
            return found
        except Exception as e:
            logger.warning(f"[{pass_name}] Tesseract error: {e}")
            return False

    # ── Semi-transparent overlay detection ────────────────────────────
    def _detect_semitransparent_overlay(self, img: np.ndarray, text_mask: np.ndarray) -> np.ndarray:
        """
        Detect the faint semi-transparent halo that surrounds watermark text.
        These are the pixels the OCR bounding box misses but that are still
        visually altered from the original image.

        Strategy: in the area around detected text, look for pixels that are
        brighter/lighter than expected (the white watermark overlay lifts
        dark pixels slightly).
        """
        h, w = img.shape[:2]
        overlay_mask = np.zeros((h, w), dtype=np.uint8)

        if text_mask.max() == 0:
            return overlay_mask

        # Create a generous search area around detected text (but not the whole image)
        search_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (80, 40))
        search_area = cv2.dilate(text_mask, search_kernel)
        # Exclude already-detected pixels
        search_only = cv2.bitwise_and(search_area, cv2.bitwise_not(text_mask))

        # Convert to LAB for better luminance comparison
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0].astype(np.float32)

        # Compute local mean luminance using a large kernel
        local_mean = cv2.GaussianBlur(l_channel, (51, 51), 0)

        # Pixels that are significantly brighter than local mean = watermark halo
        diff = l_channel - local_mean
        bright_pixels = (diff > 8) & (search_only > 0)

        overlay_mask[bright_pixels] = 255

        # Filter: only keep components that are connected/close to text mask
        # (prevent random bright spots from being included)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            overlay_mask, connectivity=8
        )
        filtered = np.zeros_like(overlay_mask)
        for label in range(1, num_labels):
            component = (labels == label).astype(np.uint8) * 255
            # Check if this component overlaps with the expanded text region
            overlap = cv2.bitwise_and(component, search_area)
            if overlap.sum() > 0 and stats[label][cv2.CC_STAT_AREA] > 50:
                filtered[labels == label] = 255

        return filtered

    # ── Logo detection ───────────────────────────────────────────────
    def _detect_nearby_logos(self, img: np.ndarray, text_mask: np.ndarray) -> np.ndarray:
        """
        Detect circular/geometric logos near detected text.
        Uses contour analysis + Hough circle detection.
        """
        h, w = img.shape[:2]
        logo_mask = np.zeros((h, w), dtype=np.uint8)

        if text_mask.max() == 0:
            return logo_mask

        # Search region: expand left and right from text
        search_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (120, 30))
        search_area = cv2.dilate(text_mask, search_kernel)
        search_only = cv2.bitwise_and(search_area, cv2.bitwise_not(text_mask))

        # Edge detection in search area
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        edges_in_search = cv2.bitwise_and(edges, search_only)

        # Find contours that look like logos (circles, rectangles, etc.)
        contours, _ = cv2.findContours(edges_in_search, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 200 or area > 30000:
                continue

            # Check circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)

            # Only accept shapes with reasonable circularity (logo-like)
            if circularity > 0.4:
                x, y, bw, bh = cv2.boundingRect(contour)
                # Fill the bounding rect with some padding
                pad = 5
                x0 = max(0, x - pad)
                y0 = max(0, y - pad)
                x1 = min(w, x + bw + pad)
                y1 = min(h, y + bh + pad)
                cv2.rectangle(logo_mask, (x0, y0), (x1, y1), 255, -1)
                logger.info(f"Logo contour: area={area}, circ={circularity:.2f}")

        # Also try Hough circle detection (strict parameters to avoid false positives)
        gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        search_gray = cv2.bitwise_and(gray_blurred, search_only)
        if search_gray.max() > 0:
            circles = cv2.HoughCircles(
                search_gray,
                cv2.HOUGH_GRADIENT,
                dp=1.5,
                minDist=50,
                param1=80,
                param2=50,
                minRadius=15,
                maxRadius=60
            )
            if circles is not None:
                # Limit to at most 3 circles to prevent over-detection
                for circle in circles[0][:3]:
                    cx, cy, radius = int(circle[0]), int(circle[1]), int(circle[2])
                    cv2.circle(logo_mask, (cx, cy), radius + 8, 255, -1)
                    logger.info(f"Hough circle detected at ({cx},{cy}) r={radius}")

        return logo_mask

    # ── Fallback center mask ─────────────────────────────────────────
    def generate_center_mask(self, img: np.ndarray, fraction: float = 0.1) -> np.ndarray:
        """
        Fallback: generate a much smaller rectangle in center if detection fails.
        """
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        mh = int(h * fraction)
        mw = int(w * (fraction * 2.0))
        y0 = (h - mh) // 2
        x0 = (w - mw) // 2
        cv2.rectangle(mask, (x0, y0), (x0 + mw, y0 + mh), 255, -1)

        # Feather significantly
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return mask

    # ── Helpers ───────────────────────────────────────────────────────
    def _center_crop_coords(
        self,
        h: int,
        w: int,
        fraction: Optional[float] = None
    ) -> Tuple[int, int, int, int]:
        frac = fraction or self.center_fraction
        mh = int(h * frac)
        mw = int(w * frac)
        y0 = (h - mh) // 2
        x0 = (w - mw) // 2
        return x0, y0, x0 + mw, y0 + mh

    def _dilate_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Aggressive closing + dilation to:
        1. Bridge gaps between characters
        2. Fill holes inside letters (e.g. 'O', 'B', 'R')
        3. Create a smooth, continuous mask for clean inpainting
        """
        # Step 1: Large closing to bridge character gaps
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)

        # Step 2: Fill any interior holes
        # (inverted flood-fill from corner, then invert back)
        filled = mask.copy()
        flood = np.zeros_like(filled)
        contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(flood, contours, -1, 255, -1)
        mask = cv2.bitwise_or(mask, flood)

        # Step 3: Dilate outward to cover any semi-transparent fringe
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.dilate(mask, dilate_kernel, iterations=2)

        # Step 4: Feather edges for smooth inpainting boundary
        mask = cv2.GaussianBlur(mask, (9, 9), 0)
        _, mask = cv2.threshold(mask, 20, 255, cv2.THRESH_BINARY)

        return mask

    def _proximity_bloom(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Final pass: search for any remaining high-contrast fragments
        near the detected watermark. Uses a wide horizontal kernel to
        catch logos positioned to the left/right of text.
        """
        if mask.max() == 0:
            return mask

        h, w = img.shape[:2]
        # wide horizontal reach, narrow vertical to avoid background damage
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 20))
        search_area = cv2.dilate(mask, kernel)

        # Detect high-frequency structures (text/logo edges)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        diff = cv2.absdiff(gray, blurred)
        _, thresh = cv2.threshold(diff, 8, 255, cv2.THRESH_BINARY)

        # Only consider structures within search area
        blobs = cv2.bitwise_and(thresh, search_area)

        # Connected component filter
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            blobs, connectivity=8
        )
        bloom_mask = mask.copy()
        for label in range(1, num_labels):
            stat = stats[label]
            area = stat[cv2.CC_STAT_AREA]
            if 100 < area < 15000:
                blob_region = (labels == label).astype(np.uint8) * 255
                # Verify it's near existing mask (within dilated region)
                overlap = cv2.bitwise_and(blob_region, search_area)
                if overlap.sum() > 0:
                    bloom_mask[labels == label] = 255

        return bloom_mask
