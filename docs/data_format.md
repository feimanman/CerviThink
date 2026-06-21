# Data Format

Input annotations are JSONL files. Each line describes one cell image.

Required fields:

```json
{"image": "IMAGE_PATH_001", "width": 256, "height": 256, "label": "HSIL", "bbox": [120, 95, 220, 210]}
```

For patient-level splits, raw private rows may include `patient_id`, `case_id`,
`slide_id`, `subject_id`, `study_id`, or `group_id`. These fields are used only
to keep groups separated across train/test and are not written to CerviCoT rows.

Supported labels:

```text
HSIL, ASC-H, LSIL, ASC-US, Normal
```

Accepted aliases include `ASCUS`, `ASC-H`, `NILM`, `negative`, `normal`, and
the full Bethesda names used in the converter.

The converted training JSONL contains:

```text
image, width, height, input_width, input_height, bboxs, problem, solution, label, split
```

`solution` follows the tagged format used for SFT:

```text
<think>...</think> <box>[x1,y1,x2,y2]</box> <rethink>...</rethink> <answer>HSIL</answer>
```

For DST and ComparisonDetector reproduction, first crop each annotated cell and
resize the crop to `256 x 256`:

```bash
python3 scripts/prepare_cell_crops.py \
  --input YOUR_ANNOTATIONS.jsonl \
  --image-root YOUR_IMAGE_ROOT \
  --output outputs/DST_cropped.jsonl \
  --output-image-dir outputs/DST_256 \
  --split-if-missing
```

For HiCervix rows that are already cell crops, use `--resize-full-image`.

The crop script writes anonymous numeric IDs by default. Use
`--keep-source-id` only for private runs where source identifiers are safe to
retain.
