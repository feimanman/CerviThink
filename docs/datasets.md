# Dataset contract for the method-repair branch

The experiment path requires an explicit patient/WSI grouping column and frozen
partitions BEFORE cropping, rationale generation, or few-shot sampling.

```json
{"image":"IMAGE_PATH","label":"HSIL","bbox":[20,20,100,100],"patient_id":"PRIVATE_GROUP","split":"train"}
```

Minimum raw rows should contain image/label/bbox, a grouping field chosen in the
configuration, and split. Width/height may be inferred from the image. Existing
partitions are checked for both group and original-image overlaps. Partial splits,
unknown splits, duplicate samples, and missing group fields cause errors.

External label mapping is an explicit JSON object, for example a source class
name mapped to one exact target label, or null for exclusion. Supply every source
label. This release does NOT provide a made-up five-class ComparisonDetector
mapping or a source of Normal cases. Confirm original task label space, Normal
provenance, and HiCervix leaf-class mapping with the original experiment records.

For external few-shot data, preserve original train/test and sample 10% only from
train. The actual sample counts, supports, seed and membership hashes are saved.
Original patient/group fields remain in the private split artifact; cropped
CerviCoT records omit them. Derived images, paths, questions and rationales still
require a separate privacy review before any publication.

Preprocessing crops the annotation-derived region (optional context padding),
resizes it to 256x256 and maps the target box into this new view. With no padding,
the target is the full crop. With padding, the target remains an inner box instead
of incorrectly becoming the full padded image. SFT uses this target; RL does not
consume it as a localization reward.

See `docs/reproduction.md` for explicit commands and protocol details. Do not
publish patient images, identifiable annotations, source paths or private configs.
