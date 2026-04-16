
import sys
sys.path.insert(0, '.')

import numpy as np
from PIL import Image
import io
import logging
import cv2
import traceback

from detection import WatermarkDetector
from inpainting import InpaintingEngine
from postprocessing import PostProcessor

logging.basicConfig(level=logging.INFO)

def test_pipeline():
    detector = WatermarkDetector()
    inpainter = InpaintingEngine()
    post_processor = PostProcessor()
    
    print(f"Subsystems: Detector={detector.ocr_backend}, Inpainter={inpainter.model_name}")
    
    # Create test image with watermark-like text
    img = np.random.randint(100, 200, (667, 1000, 3), dtype=np.uint8)
    cv2.putText(img, "NOBROKER", (300, 350), cv2.FONT_HERSHEY_SIMPLEX, 2, (230, 230, 230), 3)
    cv2.circle(img, (260, 340), 30, (230, 230, 230), 2)
    
    print(f"Test image shape: {img.shape}")

    print("\n=== Testing detection ===")
    try:
        mask = detector.detect_watermark(img)
        mask_coverage = mask.sum() / (mask.shape[0] * mask.shape[1] * 255) * 100
        print(f"Mask shape: {mask.shape}, Coverage: {mask_coverage:.2f}%")
    except Exception as e:
        print(f"Detection FAILED: {e}")
        traceback.print_exc()
        return

    print("\n=== Testing inpainting ===")
    try:
        result = inpainter.inpaint(img, mask)
        print(f"Inpainting complete. Result shape={result.shape}")
    except Exception as e:
        print(f"Inpainting FAILED: {e}")
        traceback.print_exc()
        return

    print("\n=== Testing post-processing ===")
    try:
        final = post_processor.process(result, original=img, mask=mask)
        print(f"Post-processing complete. Final shape={final.shape}")
    except Exception as e:
        print(f"Post-processing FAILED: {e}")
        traceback.print_exc()
        return

    print("\n✅ FULL PIPELINE SUCCESS")

if __name__ == "__main__":
    test_pipeline()
