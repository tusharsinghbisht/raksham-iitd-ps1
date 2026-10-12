"""Build the single submission PDF from the HTML sources.

Usage (needs pypdf, e.g. the eval venv):
    D:\\raksham_iitd\\eval\\.venv\\Scripts\\python build_pdf.py
"""
import os
import subprocess
import tempfile
from pathlib import Path

from pypdf import PdfReader, PdfWriter

HERE = Path(__file__).resolve().parent
OUT = HERE / "EchoLock_RapidActionForce_Submission.pdf"
PARTS = [
    ("EchoLock_Pitch_Deck.html", "Pitch deck (10 slides)"),
    ("EchoLock_Architecture.html", "Technical architecture (3 pages)"),
    ("EchoLock_Appendix.html", "Appendix (1 page): evidence and disclosures"),
]
EDGE = next(p for p in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe") if os.path.exists(p))

with tempfile.TemporaryDirectory() as tmp:
    w = PdfWriter()
    for i, (html, title) in enumerate(PARTS):
        pdf = Path(tmp) / f"{i}.pdf"
        subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--no-first-run", "--no-pdf-header-footer",
                        f"--user-data-dir={Path(tmp) / 'profile'}", f"--print-to-pdf={pdf}", (HERE / html).as_uri()],
                       check=True, timeout=120)
        r = PdfReader(pdf)
        start = len(w.pages)
        for page in r.pages:
            w.add_page(page)
        w.add_outline_item(title, start)
        print(f"{html}: {len(r.pages)} pages")
    w.add_metadata({"/Title": "EchoLock: RAKSHAM PS-01 submission, Team RapidActionForce (DTU)",
                    "/Author": "Team RapidActionForce, Delhi Technological University"})
    with open(OUT, "wb") as f:
        w.write(f)
print(f"{OUT.name}: {len(PdfReader(OUT).pages)} pages, {OUT.stat().st_size // 1024} KB")
