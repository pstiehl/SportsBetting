"""Programmatic SVG logo generation for Flashcat.

Orange cat face — round face, triangular ears, three whiskers per side,
black eye/nose outlines. Scalable, no rasterization.
"""

from __future__ import annotations

from pathlib import Path

ORANGE = "#F58220"
ORANGE_DARK = "#C7610B"
INK = "#1a1a1a"
PINK = "#F5A8B5"
WHITE = "#ffffff"


def cat_face_svg(size: int = 256, with_text: bool = False) -> str:
    """Return SVG XML for the Flashcat cat-face logo."""
    cx = cy = 128
    # ear paths (triangular)
    left_ear = f"M{cx - 78},{cy - 70} L{cx - 110},{cy - 130} L{cx - 32},{cy - 95} Z"
    right_ear = f"M{cx + 78},{cy - 70} L{cx + 110},{cy - 130} L{cx + 32},{cy - 95} Z"
    left_inner_ear = f"M{cx - 78},{cy - 78} L{cx - 100},{cy - 120} L{cx - 50},{cy - 92} Z"
    right_inner_ear = f"M{cx + 78},{cy - 78} L{cx + 100},{cy - 120} L{cx + 50},{cy - 92} Z"
    # whiskers
    whisker_left = "".join(
        f'<line x1="{cx - 28}" y1="{cy + 14 + i * 14}" x2="{cx - 110}" y2="{cy + 6 + i * 18}" '
        f'stroke="{INK}" stroke-width="3" stroke-linecap="round"/>'
        for i in range(3)
    )
    whisker_right = "".join(
        f'<line x1="{cx + 28}" y1="{cy + 14 + i * 14}" x2="{cx + 110}" y2="{cy + 6 + i * 18}" '
        f'stroke="{INK}" stroke-width="3" stroke-linecap="round"/>'
        for i in range(3)
    )
    text_block = (
        f'<text x="{cx}" y="{cy + 130}" text-anchor="middle" '
        f'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,system-ui,sans-serif" '
        f'font-size="34" font-weight="800" fill="{INK}" letter-spacing="-0.5">FLASHCAT</text>'
        if with_text
        else ""
    )
    view_h = 296 if with_text else 256
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 {view_h}" width="{size}" height="{int(size * view_h / 256)}">
  <!-- ears -->
  <path d="{left_ear}" fill="{ORANGE}" stroke="{INK}" stroke-width="3" stroke-linejoin="round"/>
  <path d="{right_ear}" fill="{ORANGE}" stroke="{INK}" stroke-width="3" stroke-linejoin="round"/>
  <path d="{left_inner_ear}" fill="{PINK}"/>
  <path d="{right_inner_ear}" fill="{PINK}"/>
  <!-- face circle -->
  <circle cx="{cx}" cy="{cy}" r="78" fill="{ORANGE}" stroke="{INK}" stroke-width="3"/>
  <!-- subtle orange-dark stripe across forehead -->
  <path d="M{cx - 40},{cy - 50} Q{cx},{cy - 60} {cx + 40},{cy - 50}" stroke="{ORANGE_DARK}" stroke-width="4" fill="none" stroke-linecap="round"/>
  <!-- eyes (almond, black outline, green pupils) -->
  <ellipse cx="{cx - 26}" cy="{cy - 8}" rx="13" ry="17" fill="{WHITE}" stroke="{INK}" stroke-width="3"/>
  <ellipse cx="{cx + 26}" cy="{cy - 8}" rx="13" ry="17" fill="{WHITE}" stroke="{INK}" stroke-width="3"/>
  <ellipse cx="{cx - 26}" cy="{cy - 8}" rx="4.5" ry="14" fill="{INK}"/>
  <ellipse cx="{cx + 26}" cy="{cy - 8}" rx="4.5" ry="14" fill="{INK}"/>
  <!-- nose -->
  <path d="M{cx - 8},{cy + 18} L{cx + 8},{cy + 18} L{cx},{cy + 30} Z" fill="{PINK}" stroke="{INK}" stroke-width="2.5" stroke-linejoin="round"/>
  <!-- mouth -->
  <path d="M{cx},{cy + 30} L{cx},{cy + 40}" stroke="{INK}" stroke-width="3" stroke-linecap="round"/>
  <path d="M{cx},{cy + 40} Q{cx - 12},{cy + 50} {cx - 18},{cy + 44}" stroke="{INK}" stroke-width="3" fill="none" stroke-linecap="round"/>
  <path d="M{cx},{cy + 40} Q{cx + 12},{cy + 50} {cx + 18},{cy + 44}" stroke="{INK}" stroke-width="3" fill="none" stroke-linecap="round"/>
  <!-- whiskers -->
  {whisker_left}
  {whisker_right}
  {text_block}
</svg>"""


def write_logo(path: Path, *, with_text: bool = False, size: int = 256) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cat_face_svg(size=size, with_text=with_text))


def write_favicon_png(path: Path, size: int = 32) -> None:
    """Write a simple PNG favicon via Pillow (no cairosvg required)."""
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pad = 2
    orange = (245, 130, 32, 255)
    ink = (26, 26, 26, 255)
    pink = (245, 168, 181, 255)
    # ears
    d.polygon([(pad + 2, 11), (pad + 2, 2), (12, 9)], fill=orange, outline=ink)
    d.polygon([(size - pad - 2, 11), (size - pad - 2, 2), (size - 12, 9)], fill=orange, outline=ink)
    # face
    d.ellipse([pad, 7, size - pad, size - pad], fill=orange, outline=ink)
    # eyes
    d.ellipse([10, 14, 14, 20], fill=ink)
    d.ellipse([size - 14, 14, size - 10, 20], fill=ink)
    # nose
    d.polygon([(size // 2 - 2, 22), (size // 2 + 2, 22), (size // 2, 25)], fill=pink, outline=ink)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
