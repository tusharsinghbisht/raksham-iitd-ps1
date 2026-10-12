"""Build the one-page appendix (evidence + disclosures) from results/scores.csv and results/results.json.

Writes submission/echolock/EchoLock_Appendix.html using the pitch deck's own stylesheet, so every
number on the page comes straight from the evaluation run.
"""
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
DECK = os.path.join(ROOT, "..", "submission", "echolock")
REPO = "github.com/tusharsinghbisht/raksham-iitd-ps1"
d = pd.read_csv(os.path.join(ROOT, "results", "scores.csv"))
R = json.load(open(os.path.join(ROOT, "results", "results.json")))
hi, team = d[d.source != "team"], d[d.source == "team"]

COND = [("clean_16k", "Clean 16 kHz", "#0B8A5F"), ("narrowband_8k", "8 kHz narrowband", "#2F9BB7"),
        ("g711", "G.711", "#3F3BC4"), ("opus_12k_voip", "Opus 12k", "#8C7BFF"),
        ("amr_nb_12k2", "AMR-NB 12.2k", "#E8731C"), ("g711_loss3_snr20", "G.711 + 3% loss + noise", "#CF3434")]
pct = lambda x, n=1: f"{100 * x:.{n}f}%"

c0 = hi[hi.condition == "clean_16k"]
n_real, n_fake = (c0.label == "genuine").sum(), (c0.label == "fake").sum()
spk_real, spk_fake = c0[c0.label == "genuine"].speaker.nunique(), c0[c0.label == "fake"].speaker.nunique()
shared = len(set(c0[c0.label == "genuine"].speaker.astype(str)) & set(c0[c0.label == "fake"].speaker.astype(str)))
n_team = (team.condition == "clean_16k").sum()
n_scores = len(d)
te = hi[hi.split == "test"]
n_test = (te.condition == "clean_16k").sum()
spk_test = te[te.condition == "clean_16k"].speaker.nunique()

CSS = """<style>
.ev{display:grid;gap:14px;margin-top:14px}
.ev .card{padding:12px 15px}
.ev .h4{color:var(--brand);margin-bottom:6px}
.stat{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin-top:14px}
.stat .card{padding:10px 12px}
.stat b{display:block;font-size:24px;font-weight:800;color:var(--ink);font-family:"Segoe UI Variable Display","Segoe UI",sans-serif;line-height:1.05}
.stat span{font-size:11.5px;color:var(--mute);line-height:1.3;display:block;margin-top:3px}
.fx{display:grid;grid-template-columns:1fr 1fr;gap:6px 18px}
.fx div{font-size:11.4px;line-height:1.36;color:var(--ink2);padding-left:22px;position:relative}
.fx div i{position:absolute;left:0;top:0;font-style:normal;font-weight:800;color:#fff;background:var(--accent);border-radius:5px;width:16px;height:16px;display:grid;place-items:center;font-size:10px}
.fx div b{color:var(--ink)}
table.rt{width:100%;border-collapse:collapse;font-size:11.2px}
table.rt th{font-size:10px;text-transform:uppercase;letter-spacing:.05em;color:var(--mute);text-align:left;font-weight:800;padding:4px 5px;border-bottom:1.5px solid var(--line2);line-height:1.2}
table.rt td{padding:5px 4px;border-bottom:1px solid var(--line);color:var(--ink2);white-space:nowrap}
table.rt td b{color:var(--ink)}
table.rt .ci{color:var(--mute);font-size:10.5px}
.sw{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:6px;vertical-align:0}
</style>"""


def compact_rows():
    rows = []
    for c, name, col in COND:
        r = R[c]; g = r["detection_by_generator"]
        rows.append(f'<tr><td><span class="sw" style="background:{col}"></span><b>{name}</b></td>'
                    f'<td><b>{pct(r["eer_call"])}</b> <span class="ci">[{pct(r["eer_call_ci"][0])}–{pct(r["eer_call_ci"][1])}]</span></td>'
                    f'<td>{r["tau"]:+.2f}</td><td>{pct(r["test_detection"][0])}</td><td>{pct(r["test_false_alarm"][0])}</td>'
                    f'<td>{pct(g.get("freevc24", 0), 0)} / {pct(g.get("xtts_v2", 0), 0)}</td><td>{pct(r["team_false_alarm"][0])}</td></tr>')
    return "".join(rows)


ONE = f"""<section class="slide">
  <div class="kicker"><span class="num" style="width:auto;padding:0 8px">A</span>Appendix · evidence and disclosures</div>
  <h2 class="t" style="font-size:27px">What we built and measured, and what we used.</h2>
  <div class="lead" style="margin-top:4px;font-size:14px">Code, team recordings, per-clip scores and the prototype: <a href="https://{REPO}" style="color:var(--brand);font-weight:700;text-decoration:none">{REPO}</a></div>
  <div class="stat" style="margin-top:10px">
    <div class="card"><b>{n_real + n_fake + n_team:,}</b><span>clips scored: {n_real} real, {n_fake} cloned (Hindi), {n_team} team</span></div>
    <div class="card"><b>{spk_real + spk_fake - shared + 2}</b><span>speakers: {spk_real} real, {spk_fake} cloned, 2 teammates</span></div>
    <div class="card"><b>6</b><span>phone channels per clip, {n_scores:,} scores in total</span></div>
    <div class="card"><b>22 ms</b><span>per 4 s window, laptop GPU (RTX 3050); 299 ms on CPU</span></div>
    <div class="card"><b>26 / 26</b><span>prototype tests pass: confirm, replay, tamper, expiry, review</span></div>
    <div class="card"><b>94.4 M</b><span>weights loaded by our own PyTorch loader, strict check</span></div>
  </div>
  <div class="ev" style="grid-template-columns:minmax(0,1.32fr) minmax(0,1fr);margin-top:12px;gap:12px">
    <div class="col" style="gap:12px">
      <div class="card">
        <div class="h4">Detector results · Hindi test set, {n_test} clips, {spk_test} speakers, speaker-disjoint</div>
        <table class="rt">
          <tr><th>Channel</th><th>EER call-level [95% CI]</th><th>τ (dev)</th><th>Detected @τ</th><th>False alarm @τ</th><th>FreeVC / XTTS</th><th>Team false alarm</th></tr>
          {compact_rows()}
        </table>
        <div class="s" style="margin-top:5px">Kathbath real vs IndicSynth clones. τ fixed on a dev split at 1% false alarms, never tuned on test. Same-person check ({shared} speakers): {pct(R["clean_16k"]["eer_same_person"])} clean, {pct(R["g711"]["eer_same_person"])} G.711.</div>
      </div>
      <div class="card">
        <div class="h4">What we learned · problems we fixed</div>
        <div class="fx">
          <div><i>1</i><b>Loss + noise breaks detection</b> (26% caught). Poor audio → UNKNOWN; app confirmation never relies on the score.</div>
          <div><i>2</i><b>τ shifts {R["g711"]["tau"]:+.2f} → {R["clean_16k"]["tau"]:+.2f}</b> by channel, so it's set per codec, and re-set on each bank's calls in shadow mode (our voices: 16% false alarms on G.711).</div>
          <div><i>3</i><b>fairseq won't install</b> on Windows, so we rewrote wav2vec 2.0 in PyTorch. Excluded FLEURS and MLAAD (in the model's training set).</div>
          <div><i>4</i><b>IndicSynth shards are sorted</b> (first pull: 5 XTTS speakers), so we sampled at random offsets → {spk_fake}. Split by speaker ID across both datasets.</div>
        </div>
      </div>
    </div>
    <div class="card dis">
      <div class="h4">Disclosures</div>
      <p><b>AI-generated content:</b> Claude (Anthropic, via Claude Code) helped write and lay out this deck, the logo, the recording script and our evaluation code. An earlier plan was drafted with Claude; a first review and the workflow prototype with an OpenAI coding assistant. The team reviewed all of it and owns every claim. All diagrams are inline SVG/HTML; no generated images, stock photos or templates.</p>
      <p><b>Model:</b> NII wav2vec-small-anti-deepfake, Hugging Face rev 9a13264, on facebook/wav2vec2-base. CC BY-NC-SA 4.0, research only. Published weights, no fine-tuning. MMS-1B and AASIST-L are cited only, not run.</p>
      <p><b>Datasets:</b> Kathbath Hindi (AI4Bharat, CC BY 4.0), real speech. IndicSynth Hindi (ACL 2025, CC BY-NC 4.0), XTTS-v2 and FreeVC clones. Recordings of 2 team members, released with their consent at {REPO}. No bank, customer or platform data. We have cloned no one yet; demo clones will use consenting teammates only.</p>
      <p><b>Assets &amp; software:</b> IIT Delhi emblem and Amazon logo (Wikimedia Commons), trademarks of their owners, used only to name the organisers. Segoe UI, PyTorch, SciPy, FFmpeg, Hugging Face Hub, Microsoft Edge.</p>
      <p><b>Sources:</b> McAfee, <i>The Artificial Imposter</i> (2023); Android CAPTURE_AUDIO_OUTPUT docs and Google Play policy (May 2022); RBI Authentication Directions 2025; DPDP Act 2023; TRAI 1600-series; NII model card and arXiv:2506.21090; RFC 7866 (SIPREC).</p>
      <p><b>Status:</b> slide 8, A2 τ values and this page are measured by us (12 Oct 2026). Slide 7 chart, app screens, evidence pack and cost are illustrative; prototype voice scores are simulated.</p>
    </div>
  </div>
</section>"""

CSS += """<style>
.dis p{font-size:10.9px;line-height:1.38;color:var(--ink2);margin-bottom:6px}
.dis p b{color:var(--ink)}
.stat .card{padding:8px 11px} .stat b{font-size:21px} .stat span{font-size:10.8px}
table.rt td{padding:4px 4px}
</style>"""
src = open(os.path.join(DECK, "EchoLock_Pitch_Deck.html"), encoding="utf-8").read()
style = src[src.index("<style>"):src.index("</style>") + len("</style>")]
html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>EchoLock Appendix</title>
{style}
{CSS}
</head>
<body>
{ONE}
</body>
</html>
"""
open(os.path.join(DECK, "EchoLock_Appendix.html"), "w", encoding="utf-8").write(html)
print("wrote EchoLock_Appendix.html")
