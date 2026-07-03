# PRODUCT DATA LABELER - HTML BASED Features

- **Draw boxes** by click-and-drag directly on the image
- **Manage classes** with auto-assigned colors, one per product and gives coordinates normalized (0, 1)
- **Keyboard shortcuts** for fast labeling (`1`–`9` to switch class, `F` to fit, `Ctrl+wheel` to zoom)
- **Right-click** any box to delete it
- **Export to standard ML formats:**
  - **YOLO** `.txt` (with companion `classes.txt`)
  - **COCO** `.json`
  - **Individual JPEG crops** named `<class>_001.jpg`, `<class>_002.jpg`, ...
- 💾 **Save / load sessions** as JSON to resume labeling later



## 📖 How to Use

### Step 1 — Load an image
Click **Choose File** in the sidebar and pick a shelf photo.

### Step 2 — Add product classes
Type a product identifier in `snake_case` and click **Add**:

```
zanuy_tortillachips_chia_130g
ametller_natxos_formatge_150g
natursoy_nachos_maiz_125g
leche_lletnostra_semidesnatada_1l
bebidavegetal_yosoy_avena_barista_1l
```

Each class gets a unique color automatically.

### Step 3 — Draw bounding boxes
Click and drag on the image to draw a box around each product.
The active class (highlighted in the sidebar) is what gets assigned.

Use number keys **1–9** to switch classes quickly.

### Step 4 — Export

| Button | Output | Use case |
|---|---|---|
| **YOLO .txt** | `<image>.txt` + `classes.txt` | Training YOLOv5, YOLOv8, Ultralytics |
| **COCO .json** | Standard COCO format | Detectron2, MMDetection |
| **Crops (jpg)** | Individual cropped JPEGs | Classification datasets, CLIP embeddings |
| **Save .json** | Project state | Resume labeling later |

---

## 📂 Output Formats

### YOLO (`.txt`)

One line per box:

```
0 0.512500 0.481250 0.234375 0.362500
1 0.734375 0.487500 0.203125 0.362500
0 0.281250 0.487500 0.218750 0.375000
```

Format: `class_id x_center y_center width height` — all values normalized to `[0, 1]`.

A `classes.txt` file with one class name per line is exported alongside.

### COCO (`.json`)

Standard COCO detection schema:

```json
{
  "images": [
    { "id": 1, "file_name": "shelf001.jpg", "width": 4032, "height": 3024 }
  ],
  "categories": [
    { "id": 1, "name": "zanuy_tortillachips_chia_130g", "supercategory": "product" }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [1620, 890, 780, 1100],
      "area": 858000,
      "iscrowd": 0
    }
  ]
}
```

### Crops (`.jpg`)

Each bounding box is cropped and saved as a separate JPEG:

```
zanuy_tortillachips_chia_130g_001.jpg
zanuy_tortillachips_chia_130g_002.jpg
ametller_natxos_formatge_150g_001.jpg
natursoy_nachos_maiz_125g_001.jpg
```

Instance numbers auto-increment per class — useful for building a classification or embedding dataset.

### Project (`.json`)

A full save of the current session — classes, boxes, and image metadata.
Load it back later with **Load saved .json** to keep working.

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
|---|---|
| `drag` | Draw a bounding box |
| `right-click` | Delete the box under the cursor |
| `1` – `9` | Switch to class 1–9 |
| `Ctrl` + `wheel` | Zoom in / out |
| `F` | Fit image to screen |

---

## Typical Workflow

```
Shelf photos
    │
    ▼
[ Product Labeler ]  ← this tool
    │
    ├─── YOLO .txt ──────────► Train YOLO detector
    ├─── COCO .json ─────────► Train Detectron2 / MMDetection
    └─── Crops .jpg ─────────► Build product catalog / CLIP embeddings
```

---

## Notes

- The tool does **not auto-save**. Click **Save .json** before closing the tab.
- Runs entirely client-side — no network requests, no data uploaded anywhere.
- Best used with images up to ~15 MP. Larger images work but may be sluggish.
- Tested on Chrome, Safari, and Firefox.


