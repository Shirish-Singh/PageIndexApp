"""
PageIndex Document QA - Application Launcher

This script provides a convenient way to run the Streamlit application.

Usage:
    python main.py          # Run the Streamlit app
    python main.py --help   # Show help

Or run directly with Streamlit:
    streamlit run app.py
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Launch the Streamlit application."""
    app_path = Path(__file__).parent / "app.py"

    if not app_path.exists():
        print(f"Error: app.py not found at {app_path}")
        sys.exit(1)

    # Check if streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("Streamlit is not installed. Installing dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

    # Run the Streamlit app using the current Python interpreter
    print("Starting PageIndex Document QA...")
    print("Open your browser at http://localhost:8501")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])


if __name__ == "__main__":
    main()
