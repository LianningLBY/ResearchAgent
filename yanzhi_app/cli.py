import subprocess
import sys
from pathlib import Path


def run():
    app_path = Path(__file__).parent / "app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path),
         "--server.port", "8501", "--server.headless", "true"],
        check=True,
    )
