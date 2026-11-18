"""Render docs/VoiceTrap_Manual.pdf from manual.html with headless Chrome.

    python docs/manual/build.py
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHROME = next((c for c in ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                           shutil.which("google-chrome"), shutil.which("chromium")] if c and Path(c).exists()), None)

html = (HERE / "manual.html").read_text().replace("<!--WAVE-->", (HERE / "wave.svg.part").read_text())
with tempfile.NamedTemporaryFile("w", suffix=".html", dir=HERE, delete=False) as f:
    f.write(html)
try:
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={HERE.parent / 'VoiceTrap_Manual.pdf'}", f.name], check=True,
                   stderr=subprocess.DEVNULL)
finally:
    Path(f.name).unlink()
print("wrote", HERE.parent / "VoiceTrap_Manual.pdf")
