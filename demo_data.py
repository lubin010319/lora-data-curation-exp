"""Generate original character studies for a reproducible data-quality exercise."""
import json
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "work" / "source"


def portrait(draw, cx, cy, scale, palette, pose):
    lines, hair, jacket, accent = palette

    def p(x, y):
        return cx + round(scale * (x + pose * y / 11)), cy + round(scale * y)

    def poly(points, fill):
        draw.polygon([p(x, y) for x, y in points], fill=fill, outline=lines, width=max(1, int(2 * scale)))

    def oval(box, fill, outline=lines):
        draw.ellipse((*p(*box[:2]), *p(*box[2:])), fill=fill, outline=outline, width=max(1, round(scale * 2)))

    poly([(-105, 104), (-98, 67), (-47, 40), (31, 39), (92, 64), (111, 104)], jacket)
    oval((-75, -94, 74, 68), hair)
    oval((-47, -71, 54, 57), "#ffe2ce")
    poly([(-73, -16), (-69, -61), (-39, -91), (17, -98), (66, -73),
          (72, -22), (35, -43), (44, -12), (1, -46), (-36, -22), (-37, 45)], hair)
    poly([(-28, 55), (-1, 78), (24, 55), (21, 91), (-9, 95)], "#f8f5e7")
    for eye in (-6, 29):
        oval((eye - 4, -15, eye + 6, 4), lines)
        oval((eye - 3, -13, eye + 1, -8), "#fff8ed", None)
    draw.line([p(8, 26), p(22, 25)], fill=lines, width=max(1, round(3 * scale)))
    draw.line([p(-52, -67), p(-32, -87), p(4, -90)], fill="#e2a4a7", width=max(1, round(3 * scale)))
    poly([(-68, -38), (-49, -51), (-42, -43), (-60, -29)], accent)
    poly([(-66, -22), (-47, -35), (-40, -27), (-58, -13)], accent)


def study(size, seed, mode):
    rng = random.Random(seed)
    width, height = size
    bg = ["#f8d9c7", "#e8ddf4", "#c7e0ec", "#f0e8ba", "#cfe9dc"][seed % 5]
    image = Image.new("RGB", size, bg)
    canvas = ImageDraw.Draw(image)
    canvas.ellipse((width * .16, height * .07, width * .83, height * .83),
                   fill="#ffffff", outline="#faf4e7", width=5)
    spacing = max(11, int(min(size) / 30))
    for y in range(spacing, round(height * .83), spacing):
        for x in range(spacing, width - spacing, spacing):
            if rng.random() > 0.72:
                canvas.ellipse((x, y, x + 3, y + 3), fill="#a3a1ae")
    for index in range(7):
        y = height * (0.78 + index * 0.033)
        canvas.line([(0, y), (width, y - height * .10)], fill="#fff9e6", width=max(2, width // 128))
    palette = ("#352a42", "#5b365d", "#275a78", "#f3be55")
    if mode == "crowd":
        portrait(canvas, round(width * .31), round(height * .55), min(size) / 410, palette, -0.6)
        portrait(canvas, round(width * .72), round(height * .53), min(size) / 420,
                 ("#393646", "#a14c43", "#476748", "#f7c25a"), 0.5)
    else:
        angle = [-0.7, 0.2, 0.8, -0.5, 0.5, 0.0][seed % 6]
        portrait(canvas, round(width * (.48 + .045 * math.sin(seed))), round(height * .52),
                 min(size) / (306 + (seed % 6) * 9), palette, angle)
    if mode == "watermark":
        canvas.rectangle((0, height * .78, width, height * .92), fill="#25283b")
        font = ImageFont.load_default(size=int(min(size) * .08))
        canvas.text((width / 2, height * .85), "SAMPLE / PREVIEW", fill="#fffcf1",
                    anchor="mm", font=font)
    return image


def make_demo():
    INPUT.mkdir(parents=True, exist_ok=True)
    designs = [
        ((768, 768), "ok"), ((800, 720), "watermark"), ((690, 910), "ok"),
        ((1000, 650), "crowd"), ((646, 978), "ok"), ((210, 260), "low resolution"),
        ((1010, 690), "caption conflict"), ((900, 676), "ok"), ((754, 752), "ok"),
        ((1030, 670), "ok"), ((640, 1000), "ok"), ((768, 768), "near duplicate"),
        ((1240, 395), "extreme ratio"), ((880, 710), "ok"),
    ]
    rows = []
    for i, (size, kind) in enumerate(designs, 1):
        image = study(size, i if kind != "near duplicate" else 1, kind)
        name = f"study_{i:02d}.jpg"
        image.save(INPUT / name, quality=81 if kind == "near duplicate" else 94)
        caption = "sorami, 1girl, purple hair, short hair, yellow hairclip, blue jacket, blue jacket"
        people = 2 if kind == "crowd" else 1
        if people == 2:
            caption = caption.replace("1girl", "2girls, multiple characters")
        if kind == "watermark":
            caption += ", watermark"
        if kind == "caption conflict":
            caption += ", long hair"
        rows.append({"id": f"study_{i:02d}", "file": name, "caption": caption,
                     "people": people, "watermark": kind == "watermark",
                     "fixture": kind, "license": "CC0-1.0"})
    manifest = INPUT / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(make_demo())
