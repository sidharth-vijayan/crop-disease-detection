# Changes — Leaf Segmentation & Detection Enhancement

## What was added

### 1. Bbox crop before CNN (`api.py`)
After `segment_leaf()` runs, the segmented image is now cropped tightly
to the leaf's bounding box before being resized to 380×380 for EfficientNet-B4.

**Before:** full frame (including grey background padding) was resized to 380×380.  
**After:** only the leaf region is resized, so all 380×380 pixels map to actual leaf tissue.

Affected lines in `api.py`: Stage 0b block (~line 1866), Grad-CAM call, segmentation response dict.
The API response now includes a `bbox_crop` boolean in the `segmentation` field.

---

### 2. Individual leaf detection (`test_leaf_detect.py`)
A standalone test script that detects **every individual leaf** in a plant photo
using SAM (Segment Anything Model) and produces two output images.

**How it works:**
1. SAM runs automatic mask generation on the full image
2. Masks are filtered by size (0.5–80% of image) and greenness (≥25% green HSV pixels)
3. Each surviving mask = one detected leaf
4. SAM's `predicted_iou` score is used as the confidence metric

**Outputs:**
- `<image>_detected.png` — original photo with a coloured bounding box and confidence score per leaf
- `<image>_hierarchy.png` — grid of cropped leaf thumbnails ranked by confidence (highest top-left)

**To run:**
```bash
# From crop-disease-detection/
python test_leaf_detect.py your_photo.jpg
```

**Dependencies:**
```bash
pip install segment-anything
# Download checkpoint (~358MB) into crop-disease-detection/
Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" -OutFile "sam_vit_b.pth"
```

**Note:** `test_leaf_detect.py` and `sam_vit_b.pth` are for local testing only.
The leaf detection pipeline using SAM is intended to be integrated into `api.py`
as a preprocessing step before the CNN — replacing or augmenting the existing
`leafseg.py` (HSV + GrabCut) approach for multi-leaf images.

---

## Files changed
| File | Change |
|---|---|
| `api.py` | Added Stage 0b — bbox crop between segmentation and CNN |
| `test_leaf_detect.py` | New — SAM-based individual leaf detection test script |
| `CHANGES.md` | New — this file |

## Files NOT to commit
| File | Reason |
|---|---|
| `sam_vit_b.pth` | 358MB model checkpoint — too large for git |
| `plant.jpg` | Test image |
| `plant_detected.png` | Generated output |
| `plant_hierarchy.png` | Generated output |
