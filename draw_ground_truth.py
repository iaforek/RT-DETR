#!/usr/bin/env python3
"""
Draw YOLO-format ground-truth bounding boxes on an image.

Expected label format:
    class_id x_center y_center width height

Coordinates are normalized to [0, 1].

If --label is omitted, the label path is derived by replacing
/images/ with /labels/ and changing the suffix to .txt.

The box and label colours intentionally match validate.py:
the colour is selected from the same palette using class_id % len(palette).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont


# Same class-colour palette used by validate.py.
CLASS_COLOURS = (
    (230, 57, 70),
    (29, 53, 87),
    (69, 123, 157),
    (42, 157, 143),
    (233, 196, 106),
    (244, 162, 97),
    (231, 111, 81),
    (131, 56, 236),
    (0, 150, 136),
    (255, 111, 0),
)


def derive_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)

    try:
        images_index = parts.index("images")
    except ValueError as exc:
        raise ValueError(
            "Could not derive label path automatically because the image path "
            "does not contain an 'images' directory. Pass --label explicitly."
        ) from exc

    parts[images_index] = "labels"
    return Path(*parts).with_suffix(".txt")


def load_class_names(path: Path | None) -> List[str] | None:
    if path is None:
        return None

    if not path.is_file():
        raise FileNotFoundError(f"Class names file not found: {path}")

    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def class_label(class_id: int, names: List[str] | None) -> str:
    if names is not None and 0 <= class_id < len(names):
        return names[class_id]
    return f"class_{class_id}"


def class_colour(class_id: int) -> Tuple[int, int, int]:
    """Return the same RGB class colour used by validate.py."""
    return CLASS_COLOURS[class_id % len(CLASS_COLOURS)]


def yolo_to_xyxy(
    x_center: float,
    y_center: float,
    box_width: float,
    box_height: float,
    image_width: int,
    image_height: int,
) -> Tuple[int, int, int, int]:
    x_center *= image_width
    y_center *= image_height
    box_width *= image_width
    box_height *= image_height

    x1 = int(round(x_center - box_width / 2.0))
    y1 = int(round(y_center - box_height / 2.0))
    x2 = int(round(x_center + box_width / 2.0))
    y2 = int(round(y_center + box_height / 2.0))

    x1 = max(0, min(image_width - 1, x1))
    y1 = max(0, min(image_height - 1, y1))
    x2 = max(0, min(image_width - 1, x2))
    y2 = max(0, min(image_height - 1, y2))

    return x1, y1, x2, y2


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw YOLO ground-truth annotations on an image."
    )
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument(
        "--label",
        default=None,
        help=(
            "Optional YOLO label file. If omitted, replace /images/ with "
            "/labels/ in the image path and use the same stem."
        ),
    )
    parser.add_argument(
        "--class-names",
        default=None,
        help="Optional file containing one class name per line.",
    )
    parser.add_argument(
        "--out-path",
        default=None,
        help="Output path. Default: <image_stem>_ground_truth.<suffix>",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    label_path = Path(args.label) if args.label else derive_label_path(image_path)
    if not label_path.is_file():
        raise FileNotFoundError(
            f"Label file not found: {label_path}\n"
            "If labels are stored elsewhere, pass --label explicitly."
        )

    names_path = Path(args.class_names) if args.class_names else None
    class_names = load_class_names(names_path)

    original = Image.open(image_path).convert("RGB")
    annotated = original.copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()

    image_width, image_height = annotated.size

    # Exact line-width rule from validate.py.
    line_width = max(2, round(min(annotated.size) / 250))

    annotations = 0

    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue

        fields = line.split()
        if len(fields) != 5:
            raise ValueError(
                f"{label_path}:{line_number}: expected 5 fields "
                f"(class x_center y_center width height), got {len(fields)}"
            )

        class_id = int(float(fields[0]))
        x_center, y_center, box_width, box_height = map(float, fields[1:])

        box = yolo_to_xyxy(
            x_center,
            y_center,
            box_width,
            box_height,
            image_width,
            image_height,
        )

        colour = class_colour(class_id)
        label = class_label(class_id, class_names)

        # Same rectangle drawing as validate.py.
        draw.rectangle(box, outline=colour, width=line_width)

        # Same label-background placement as validate.py.
        text_box = draw.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]

        label_x = max(0, box[0])
        label_y = max(0, box[1] - text_height - 6)

        draw.rectangle(
            (
                label_x,
                label_y,
                label_x + text_width + 6,
                label_y + text_height + 6,
            ),
            fill=colour,
        )
        draw.text(
            (label_x + 3, label_y + 3),
            label,
            fill="white",
            font=font,
        )

        annotations += 1

    out_path = (
        Path(args.out_path)
        if args.out_path
        else image_path.with_name(
            f"{image_path.stem}_ground_truth{image_path.suffix}"
        )
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(out_path)

    print(f"Image              : {image_path}")
    print(f"Original size      : {image_height} x {image_width}")
    print(f"Label file         : {label_path}")
    print(f"Annotations drawn  : {annotations}")
    if names_path is not None:
        print(f"Class names        : {names_path}")
    print(f"Annotated image    : {out_path}")


if __name__ == "__main__":
    main()
