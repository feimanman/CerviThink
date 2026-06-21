# Dataset Preparation

CerviThink expects one JSONL row per cervical cell image or cell annotation.
The repository does not include medical images or annotations.

## Label Space

The released code uses five Bethesda-style labels:

```text
HSIL, ASC-H, LSIL, ASC-US, Normal
```

Rows with aliases such as `ASCUS`, `NILM`, `negative`, and the full Bethesda
names are normalized during conversion.

## Input Schema

Minimum raw annotation row:

```json
{"image": "IMAGE_PATH_001", "label": "HSIL", "bbox": [120, 95, 220, 210]}
```

Recommended fields:

```json
{
  "id": "optional-source-id",
  "image": "IMAGE_PATH_001",
  "width": 1024,
  "height": 1024,
  "label": "HSIL",
  "bbox": [120, 95, 220, 210],
  "split": "train"
}
```

For patient-level splitting, include one private grouping field such as
`patient_id`, `case_id`, `slide_id`, `subject_id`, `study_id`, or `group_id` in
the raw annotation file. The preparation scripts consume it for splitting but do
not emit it in prepared CerviCoT records.

## DST

DST is treated as a private in-house dataset. Do not publish DST images,
patient identifiers, hospital identifiers, or raw annotations unless the
required approvals and data-use agreements explicitly allow it.

Use:

```bash
python3 scripts/prepare_cell_crops.py \
  --input DST_ANNOTATIONS.jsonl \
  --image-root DST_IMAGE_ROOT \
  --output outputs/reproduce/DST_cropped.jsonl \
  --output-image-dir outputs/reproduce/DST_256 \
  --dataset DST \
  --split-if-missing
```

By default, `prepare_cell_crops.py` anonymizes output IDs and crop filenames.
Use `--keep-source-id` only in private runs.

## ComparisonDetector

Prepare a JSONL file with image paths, five-class labels, and pathologist
bounding boxes. Then run the same crop command with
`--dataset ComparisonDetector`.

If the original labels are more granular than the five CerviThink labels,
perform the mapping before conversion and keep the mapping script in your
private experiment record.

## HiCervix

HiCervix contains more categories than the paper label space. Filter or map
rows to:

```text
ASC-US, ASC-H, LSIL, HSIL, Normal
```

If HiCervix rows are already cell-level crops, resize the full image:

```bash
python3 scripts/prepare_cell_crops.py \
  --input HICERVIX_ANNOTATIONS.jsonl \
  --image-root HICERVIX_IMAGE_ROOT \
  --output outputs/reproduce/HiCervix_cropped.jsonl \
  --output-image-dir outputs/reproduce/HiCervix_256 \
  --dataset HiCervix \
  --resize-full-image \
  --split-if-missing
```

## Privacy Checklist

Before publishing any derived JSONL:

```bash
python3 scripts/release_check.py
python3 scripts/open_source_check.py
```

Also inspect image paths manually. They should not include patient names,
hospital numbers, phone numbers, national IDs, or internal mount paths.
