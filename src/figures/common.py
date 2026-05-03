from __future__ import annotations

import base64
import io
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.cm
import matplotlib.colors
from PIL import Image

DHDT_VMIN = 2
DHDT_VMAX = 1
DHDT_COLORS = [(-DHDT_VMIN, "#400912"), (-0.75 * DHDT_VMIN, "#630d1c"), (-0.40 * DHDT_VMIN, "#cd721c"), (-0., "#eeeeec"), (DHDT_VMAX, "#6497e3")]
DHDT_NORMALIZER = matplotlib.colors.Normalize(vmin=-DHDT_VMIN, vmax=DHDT_VMAX, clip=True)
DHDT_SM = matplotlib.cm.ScalarMappable(norm=DHDT_NORMALIZER, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("dhdt", [(DHDT_NORMALIZER(a), b) for a, b in DHDT_COLORS]))


def svg_png_to_jpg(svg_path: Path | str, out_path: Path | str | None = None, *, quality=90):
    svg_ns = "http://www.w3.org/2000/svg"
    xlink_ns = "http://www.w3.org/1999/xlink"
    ET.register_namespace("", svg_ns)
    ET.register_namespace("xlink", xlink_ns)
    out_path = svg_path if out_path is None else out_path
    tree = ET.parse(svg_path)
    root = tree.getroot()
    for img in root.findall(f".//{{{svg_ns}}}image"):
        href = img.get(f"{{{xlink_ns}}}href")
        if not href or not href.startswith("data:image/png;base64,"):
            continue
        data = base64.b64decode(href.split(",", 1)[1])
        im = Image.open(io.BytesIO(data))
        if im.mode == "RGBA":
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        else:
            im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=quality, subsampling=0)
        img.set(f"{{{xlink_ns}}}href", "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii"))
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
