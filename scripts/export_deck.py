"""Turn an exported Slides deck into one standalone HTML page.

    python scripts/export_deck.py docs/presentation

Expects, in that directory, what a Slides artifact holds: deck.json (title,
order, faces), slides/<id>.html (one <section> each, 1920x1080, inline styles)
and assets/<asset id>.<ext> (the images and videos the slides reference as
/_blob/<asset id>). Writes index.html next to them:

- every slide in deck order, scaled to the page width, with its speaker notes
  (the slide's <aside>) underneath;
- /_blob/<id> rewritten to assets/<id>.<ext>; an <img data-video=...> becomes a
  looping, muted <video> with the image as its poster;
- <x-connector x1 y1 x2 y2> becomes an SVG arrow; a connector without
  coordinates (a flex-row spacer) becomes a centred arrow glyph.

No build tools, no network except Google Fonts.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path


def _asset_map(assets: Path) -> dict[str, str]:
    return {p.stem: f"assets/{p.name}" for p in assets.iterdir() if p.is_file()}


def _connector(match: re.Match) -> str:
    attrs = dict(re.findall(r'(\w[\w-]*)="([^"]*)"', match.group(1)))
    style = attrs.get("style", "")
    colour = re.search(r"color:\s*(#[0-9A-Fa-f]{3,8})", style)
    colour = colour.group(1) if colour else "#4A5260"
    width = re.search(r"border-width:\s*(\d+)px", style)
    width = width.group(1) if width else "3"
    dashed = "border-style:dashed" in style.replace(" ", "")
    if not {"x1", "y1", "x2", "y2"} <= attrs.keys():
        return (
            f'<div style="{html.escape(style)}; display:flex; align-items:center; '
            f'justify-content:center; font-size:40px; color:{colour}">&#8594;</div>'
        )
    x1, y1, x2, y2 = (float(attrs[k]) for k in ("x1", "y1", "x2", "y2"))
    both = attrs.get("head") == "both"
    mid = f"m{abs(hash(match.group(0))) % 10**8}"
    dash = ' stroke-dasharray="10 8"' if dashed else ""
    return (
        '<svg style="position:absolute; left:0; top:0; overflow:visible; pointer-events:none" '
        'width="1" height="1">'
        f'<defs><marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
        f'markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
        f'fill="{colour}"/></marker></defs>'
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{colour}" '
        f'stroke-width="{width}"{dash} marker-end="url(#{mid})"'
        + (f' marker-start="url(#{mid})"' if both else "")
        + "/></svg>"
    )


def _slide(text: str, assets: dict[str, str]) -> tuple[str, str]:
    notes = ""
    m = re.search(r"<aside>(.*?)</aside>", text, re.S)
    if m:
        notes = m.group(1).strip()
        text = text[: m.start()] + text[m.end() :]

    def blob(mm: re.Match) -> str:
        return assets.get(mm.group(1), mm.group(0))

    text = re.sub(r"/_blob/([0-9a-f]{32})", blob, text)
    text = re.sub(
        r'<img([^>]*?)src="([^"]+)"([^>]*?)data-video="([^"]+)"([^>]*)>',
        r'<video\1poster="\2"\3src="\4"\5 autoplay muted loop playsinline></video>',
        text,
    )
    text = re.sub(r"<x-connector([^>]*)>\s*</x-connector>", _connector, text)
    # the section is the positioning context for pinned children
    text = re.sub(
        r'<section([^>]*)style="',
        r'<section\1style="position:relative; width:1920px; '
        r"height:1080px; box-sizing:border-box; overflow:hidden; ",
        text,
        count=1,
    )
    return text, notes


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("deck", type=Path, help="directory with deck.json, slides/, assets/")
    args = ap.parse_args()

    deck = json.loads((args.deck / "deck.json").read_text())
    assets = _asset_map(args.deck / "assets")
    fonts = "\n".join(
        f'<link rel="stylesheet" href="{f["href"]}">'
        for f in deck.get("faces", {}).values()
        if f.get("href")
    )
    parts = []
    for n, sid in enumerate(deck["order"], 1):
        body, notes = _slide((args.deck / "slides" / f"{sid}.html").read_text(), assets)
        parts.append(
            f'<figure class="slide" id="{sid}"><div class="stage"><div class="canvas">{body}'
            f"</div></div><figcaption><b>{n} / {len(deck['order'])}</b> {html.escape(notes)}"
            "</figcaption></figure>"
        )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(deck.get("title", "Deck"))}</title>
{fonts}
<style>
  body {{ margin:0; background:#0d1117; color:#c9d1d9; font:15px/1.5 'IBM Plex Sans', Arial,
    sans-serif; }}
  header {{ max-width:1200px; margin:0 auto; padding:32px 16px 8px; }}
  header h1 {{ font-family:'Space Grotesk', Arial, sans-serif; margin:0 0 4px; color:#f3f1ec; }}
  main {{ max-width:1200px; margin:0 auto; padding:0 16px 64px; display:flex;
    flex-direction:column; gap:40px; }}
  .slide {{ margin:0; }}
  .stage {{ position:relative; width:100%; aspect-ratio:16/9; overflow:hidden;
    border-radius:8px; box-shadow:0 4px 24px rgba(0,0,0,.4); }}
  .canvas {{ position:absolute; left:0; top:0; width:1920px; height:1080px;
    transform-origin:0 0; }}
  .canvas p, .canvas h1, .canvas h2, .canvas h3, .canvas ul, .canvas ol {{ margin:0; }}
  .canvas table {{ border-collapse:collapse; }}
  .canvas th, .canvas td {{ text-align:left; padding:8px 12px; border-bottom:1px solid #DDD8CE; }}
  figcaption {{ font-size:14px; color:#8b949e; margin-top:10px; }}
  figcaption b {{ color:#c9d1d9; }}
</style></head><body>
<header><h1>{html.escape(deck.get("title", ""))}</h1>
<p>{len(deck["order"])} slides, exported from the Slides artifact; notes under each.</p>
</header>
<main>
{chr(10).join(parts)}
</main>
<script>
  function fit() {{
    document.querySelectorAll('.stage').forEach(function (s) {{
      s.firstElementChild.style.transform = 'scale(' + s.clientWidth / 1920 + ')';
    }});
  }}
  window.addEventListener('resize', fit); fit();
</script>
</body></html>
"""
    (args.deck / "index.html").write_text(page)
    print(f"wrote {args.deck / 'index.html'}: {len(deck['order'])} slides, {len(assets)} assets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
