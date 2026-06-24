#!/usr/bin/env python3
"""生成同花顺风格应用 Logo（桌面 favicon + iOS 图标）。"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
ACCENT = (233, 48, 48)
ACCENT_DARK = (196, 30, 30)
WHITE = (255, 255, 255)


def _lerp(a: int, b: int, t: float) -> int:
    return int(a + (b - a) * t)


def _gradient_bg(size: int) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    px = img.load()
    for y in range(size):
        t = y / max(size - 1, 1)
        r = _lerp(ACCENT[0], ACCENT_DARK[0], t * 0.6)
        g = _lerp(ACCENT[1], ACCENT_DARK[1], t * 0.6)
        b = _lerp(ACCENT[2], ACCENT_DARK[2], t * 0.6)
        for x in range(size):
            px[x, y] = (r, g, b, 255)
    return img


def _rounded_mask(size: int, radius: float) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, size - 1, size - 1), radius=radius, fill=255)
    return mask


def draw_logo(size: int = 1024) -> Image.Image:
    """绘制 Logo：红底 + 白色 K 线与趋势箭头。"""
    img = _gradient_bg(size)
    draw = ImageDraw.Draw(img)
    s = size / 1024.0

    # 白色 K 线（三根）
    candles = [
        (320, 620, 380, 480, 350, True),
        (460, 580, 520, 420, 490, True),
        (600, 520, 660, 360, 630, True),
    ]
    for x1, y1, x2, y2, xm, bullish in candles:
        cx = int(xm * s)
        top = int(min(y1, y2) * s)
        bot = int(max(y1, y2) * s)
        body_w = int(44 * s)
        draw.line([(cx, top), (cx, bot)], fill=WHITE, width=max(2, int(5 * s)))
        if bullish:
            draw.rectangle(
                [cx - body_w // 2, int(y2 * s), cx + body_w // 2, int(y1 * s)],
                fill=WHITE,
            )
        else:
            draw.rectangle(
                [cx - body_w // 2, int(y1 * s), cx + body_w // 2, int(y2 * s)],
                fill=WHITE,
            )

    # 上升趋势折线
    points = [(260, 680), (420, 560), (560, 480), (720, 340), (820, 280)]
    scaled = [(int(x * s), int(y * s)) for x, y in points]
    draw.line(scaled, fill=WHITE, width=max(3, int(10 * s)), joint="curve")

    # 箭头
    ax, ay = scaled[-1]
    aw = int(36 * s)
    ah = int(28 * s)
    draw.polygon([(ax, ay - ah), (ax + aw, ay + ah // 2), (ax - aw // 3, ay + ah // 2)], fill=WHITE)

    # 底部网格线（装饰）
    grid_y = int(740 * s)
    for gx in range(int(200 * s), int(860 * s), int(80 * s)):
        draw.line([(gx, grid_y), (gx, int(780 * s))], fill=(255, 255, 255, 60), width=max(1, int(2 * s)))

    radius = int(size * 0.185)
    mask = _rounded_mask(size, radius)
    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    out.paste(img, (0, 0), mask)
    return out


def save_png(img: Image.Image, path: Path, *, opaque_bg: tuple[int, int, int] | None = None) -> None:
    """保存 PNG。AppLogo 用透明底；AppIcon 用全幅红色底（避免四角黑边）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if opaque_bg is not None:
        bg = Image.new("RGBA", img.size, (*opaque_bg, 255))
        bg.paste(img, (0, 0), img)
        bg.convert("RGB").save(path, "PNG", optimize=True)
    else:
        img.save(path, "PNG", optimize=True)
    print(f"  ✓ {path}")


def main() -> None:
    print("生成同花顺风格 Logo …")
    logo_1024 = draw_logo(1024)
    logo_512 = logo_1024.resize((512, 512), Image.Resampling.LANCZOS)
    logo_256 = logo_1024.resize((256, 256), Image.Resampling.LANCZOS)

    save_png(logo_256, ROOT / "assets" / "icon.png", opaque_bg=ACCENT)
    save_png(logo_512, ROOT / "assets" / "icon-512.png", opaque_bg=ACCENT)
    save_png(logo_1024, ROOT / "ios" / "Sources" / "Assets.xcassets" / "AppIcon.appiconset" / "icon-1024.png", opaque_bg=ACCENT)
    save_png(logo_256, ROOT / "ios" / "Sources" / "Assets.xcassets" / "AppLogo.imageset" / "logo.png")
    save_png(logo_256, ROOT / "ios" / "Sources" / "Assets.xcassets" / "AppLogo.imageset" / "logo@2x.png")
    logo_384 = logo_1024.resize((384, 384), Image.Resampling.LANCZOS)
    save_png(logo_384, ROOT / "ios" / "Sources" / "Assets.xcassets" / "AppLogo.imageset" / "logo@3x.png")
    print("完成。")


if __name__ == "__main__":
    main()
