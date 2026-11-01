# Run the published player-comparison demo

The screenshot corresponds to Figure 4 of the author's published article:
https://offpolicyinc.com/work/claude-tennis-coach/#figure-4

A local snapshot of the public page and its site CSS/JS is saved under
`data/published-demo/` (excluded from git). The original HTML, attribution and
embedded precomputed data are retained. Local resource links were adjusted so
navigation still points to the original site. The author's historical Claude
text is existing published content, not a new API call or a Codex-generated result.

Start the saved page from any directory:

```bash
python3 -m http.server 8767 --bind 127.0.0.1 \
  --directory /Users/krum5539/Documents/deep-sports-analysis/data/published-demo
```

Open http://127.0.0.1:8767/#figure-4 . If the existing preview server is still
running, just open that URL; do not start a second server on the same port.

- Click a heatmap square to select a player and trait.
- Use 1–5 to switch that player's five example strokes.
- Use Compare with to choose Average or another player.
- Video and skeleton panels update together.
- Red/blue encode larger/smaller values, not better/worse technique.

This viewer needs no GPU, model installation, Claude API or Codex invocation.
It displays the author's already-computed results. Example videos still stream
from `media.offpolicyinc.com`, and web fonts are remote: internet access is
required. This is not a fully offline video bundle.

Browser verification: clicked Paul → Forehand → shoulders open at finish,
changed Compare with to Zverev, selected example 2; both video elements loaded
with readyState 4, the text and skeleton panel updated. Screenshot:
`data/published-demo/verified-preview.png`.

## Regenerate the comparison from source

The report generator is `dsa.pages.profiles_report`, with interactive panels in
`dsa.pages.explainer`. It requires processed reference matches and per-stroke
data, which are not included in the source checkout:

```text
data/projects/atp_hardcourt/
  profiles/{serve,forehand,backhand}.json
  profiles/{serve,forehand,backhand}_strokes.npz
  technique/{serve,forehand,backhand}/{hits.json,hits.npz}
  <match>/{source.mp4 or kept.mp4,raw.npz,players.json,names.json,strokes/}
```

Once those inputs have been generated, the upstream entry points are:

```bash
PYTHONPATH=src .venv/bin/python -m dsa.measure.shots atp_hardcourt
PYTHONPATH=src .venv/bin/python -m dsa.measure.technique extract atp_hardcourt forehand
PYTHONPATH=src .venv/bin/python -m dsa.measure.technique extract atp_hardcourt backhand
PYTHONPATH=src .venv/bin/python -m dsa.measure.technique extract atp_hardcourt serve
PYTHONPATH=src .venv/bin/python -m dsa.measure.profiles atp_hardcourt
PYTHONPATH=src .venv/bin/python -m dsa.pages.profiles_report atp_hardcourt
PYTHONPATH=src .venv/bin/python -m dsa.pages.serve data/projects/atp_hardcourt/review/profiles 8768
```

This path computes a new comparison dataset. The previously completed 23-second
demo validates the single-video pipeline; it does not supply the eleven-pro
reference dataset needed for this heatmap. New LLM work in this checkout uses
Codex CLI through `dsa.codex`, not Anthropic.
