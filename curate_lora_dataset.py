"""Build raw and curated LoRA datasets, audit every decision and plot the difference."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re

import imagehash
import numpy as np
from PIL import Image, ImageOps

from demo_data import ROOT, make_demo

WORK = ROOT / "work"
ASSETS = ROOT / "assets"
BLUE, GREEN, RED, PAPER = "#4663c5", "#10796a", "#a73e52", "#f5f6fb"
RATIOS = ((2, 3), (3, 4), (4, 5), (1, 1), (5, 4), (4, 3), (3, 2))
CONFLICTS = (("long hair", "short hair"), ("black hair", "white hair"), ("1girl", "2girls"))
WATERMARK_TAGS = {"watermark", "signature", "artist name", "logo"}
MULTI_TAGS = {"multiple characters", "multiple people", "group", "crowd"}


def bucket_for(size, resolution):
    """Nearest log aspect ratio among target-area buckets, rounded to 64 px."""
    width, height = size
    a, b = min(RATIOS, key=lambda pair: abs(math.log((width / height) / (pair[0] / pair[1]))))
    side = lambda value: max(64, round(value / 64) * 64)
    return side(resolution * math.sqrt(a / b)), side(resolution * math.sqrt(b / a))


def clean_caption(caption):
    tags = list(dict.fromkeys(re.sub(r"[_\s]+", " ", part.strip().lower())
                             for part in caption.split(",") if part.strip()))
    conflict = any(a in tags and b in tags for a, b in CONFLICTS)
    watermark = any(tag in WATERMARK_TAGS for tag in tags)
    multi = any(tag in MULTI_TAGS or re.fullmatch(r"[2-9]\d*(girls|boys|people)", tag) for tag in tags)
    return [tag for tag in tags if tag not in WATERMARK_TAGS], conflict, watermark, multi


def load_image(path):
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def plot_assets(audit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.facecolor": PAPER, "axes.facecolor": PAPER, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 6, figsize=(15.5, 7.5))
    fig.subplots_adjust(left=.025, right=.98, top=.78, bottom=.055, wspace=.12, hspace=.61)
    fig.text(.025, .94, "Same character, different dataset quality", weight="bold", fontsize=20)
    fig.text(.025, .885, "Original synthetic character studies  /  fixed QA rules and geometric buckets", fontsize=11)
    fig.text(.025, .849, "GROUP A  /  RAW", weight="bold", color=BLUE)
    fig.text(.025, .40, "GROUP B  /  CURATED & RESIZED", weight="bold", color=GREEN)
    raw = [row for row in audit if row["raw"]]
    bad = [row for row in raw if row["reasons"]]
    before = (bad + [row for row in raw if not row["reasons"]])[:6]
    after = [row for row in raw if row["curated"]][:6]
    for row_index, examples in enumerate((before, after)):
        for index, ax in enumerate(axes[row_index]):
            ax.axis("off")
            if index >= len(examples):
                continue
            row = examples[index]
            im = load_image(Path(row["raw"] if row_index == 0 else row["curated"]))
            ax.imshow(im)
            label = ", ".join(row["reasons"]) if row_index == 0 and row["reasons"] else row["id"]
            ax.set_title(f"{label}\n{im.width} × {im.height}", fontsize=8.9)
    fig.savefig(ASSETS / "curation_before_after.png", dpi=145)
    plt.close(fig)

    fig, (buckets, decisions) = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained",
                                            gridspec_kw={"width_ratios": [1.5, 1]})
    before_counts = Counter(tuple(row["preview_bucket"]) for row in raw)
    after_counts = Counter(tuple(row["preview_bucket"]) for row in raw if row["curated"])
    keys = sorted(before_counts, key=lambda size: size[0] / size[1])
    x = np.arange(len(keys))
    a = buckets.bar(x - .19, [before_counts[k] for k in keys], .36, color=BLUE, label="A raw")
    b = buckets.bar(x + .19, [after_counts[k] for k in keys], .36, color=GREEN, label="B curated")
    buckets.bar_label(a, padding=2)
    buckets.bar_label(b, padding=2)
    buckets.set_xticks(x, [f"{w}×{h}" for w, h in keys], rotation=35, ha="right")
    buckets.set_ylabel("Images")
    buckets.set_ylim(0, max(before_counts.values()) + 2)
    buckets.set_title("SDXL / ~1024 px target area", loc="left", weight="bold")
    buckets.legend(frameon=False)
    reasons = Counter(reason for row in audit for reason in row["reasons"])
    names = ["Raw readable", "Curated kept", *reasons]
    values = [len(raw), sum(after_counts.values()), *reasons.values()]
    bars = decisions.barh(names, values, color=[BLUE, GREEN] + [RED] * len(reasons))
    decisions.bar_label(bars, padding=4)
    decisions.set_xlim(0, max(values) + 3)
    decisions.set_title("Samples & rejection reasons", loc="left", weight="bold")
    decisions.invert_yaxis()
    decisions.tick_params(axis="y", length=0)
    fig.suptitle("Aspect-ratio buckets and data-quality checks", x=.02, ha="left", weight="bold", fontsize=17)
    fig.savefig(ASSETS / "aspect_bucketing_stats.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="A JSONL of reviewed source images; default generates original examples")
    args = parser.parse_args()
    manifest = (args.manifest or make_demo()).resolve()
    source = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    if not source:
        parser.error("Manifest is empty")
    identifiers = [row["id"] for row in source]
    if len(set(identifiers)) != len(identifiers) or any(not re.fullmatch(r"[\w-]+", x) for x in identifiers):
        parser.error("IDs must be unique and contain letters, digits, underscores or hyphens")
    # A new run writes the same named outputs for matching IDs. Training uses the
    # current manifests exclusively, so stale files cannot silently change a run.
    folders = {key: WORK / key for key in ("group_a", "group_b")}
    for folder in [*folders.values(), ASSETS]:
        folder.mkdir(parents=True, exist_ok=True)
    audit, reference_hashes = [], []
    for row in source:
        uid, caption = row["id"], row["caption"]
        entry = {"id": uid, "fixture": row.get("fixture"), "reasons": [], "raw": None, "curated": None}
        audit.append(entry)
        reasons = entry["reasons"]
        try:
            image = load_image(manifest.parent / row["file"])
        except (OSError, Image.DecompressionBombError):
            reasons.append("unreadable")
            continue
        path = folders["group_a"] / f"{uid}.jpg"
        image.save(path, quality=94)
        path.with_suffix(".txt").write_text(caption, encoding="utf-8")
        entry.update(raw=str(path), source_size=list(image.size),
                     preview_bucket=bucket_for(image.size, 1024), train_bucket=bucket_for(image.size, 512))
        tags, conflict, watermark, multi = clean_caption(caption)
        if min(image.size) < 384:
            reasons.append("low resolution")
        if max(image.size) / min(image.size) > 2.25:
            reasons.append("extreme ratio")
        if row.get("people", 1) > 1 or multi:
            reasons.append("multi-character")
        if row.get("watermark", False) or watermark:
            reasons.append("watermark")
        if conflict:
            reasons.append("caption conflict")
        if not tags:
            reasons.append("empty caption")
        fingerprint = imagehash.phash(image)
        thumbnail = np.asarray(image.resize((64, 64)), dtype=np.float32) / 255
        # pHash can group different poses with the same silhouette. Compare
        # thumbnail pixels as a second check before dropping a training example.
        similar = next((old_id for old, old_id, pixels in reference_hashes
                        if fingerprint - old <= 5 and np.abs(thumbnail - pixels).mean() <= .035), None)
        if not reasons and similar is not None:
            reasons.append("near duplicate")
            entry["duplicate_of"] = similar
        if reasons:
            continue
        reference_hashes.append((fingerprint, uid, thumbnail))
        path = folders["group_b"] / f"{uid}.jpg"
        ImageOps.fit(image, entry["train_bucket"], Image.Resampling.LANCZOS).save(path, quality=94)
        path.with_suffix(".txt").write_text(", ".join(tags), encoding="utf-8")
        entry["curated"] = str(path)

    for key, field in (("group_a", "raw"), ("group_b", "curated")):
        rows = [{"id": row["id"], "file": Path(row[field]).name} for row in audit if row[field]]
        (folders[key] / "manifest.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    if not any(row["curated"] for row in audit):
        raise SystemExit("Group B is empty; inspect the source manifest and QA flags.")
    (WORK / "curation_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    plot_assets(audit)
    print(json.dumps({"group_a": sum(bool(row["raw"]) for row in audit),
                      "group_b": sum(bool(row["curated"]) for row in audit),
                      "reasons": Counter(reason for row in audit for reason in row["reasons"])}, indent=2))


if __name__ == "__main__":
    main()
