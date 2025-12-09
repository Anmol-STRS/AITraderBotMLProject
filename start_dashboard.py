"""
Start both Flask backend and React frontend together
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def check_flask_installed():
    """Check if Flask is installed"""
    try:
        import flask
        return True
    except ImportError:
        return False

def main():
    """Start both servers"""
    print("=" * 70)
    print("Starting AI Trading Dashboard")
    print("=" * 70)

    # Check Flask installation
    if not check_flask_installed():
        print("\n❌ Flask is not installed!")
        print("\n📦 Please install Python dependencies first:")
        print("   pip install flask flask-cors flask-socketio python-socketio eventlet")
        print("\n   Or install all requirements:")
        print("   pip install -r requirements.txt")
        print("\n" + "=" * 70)
        input("Press Enter to exit...")
        return

    # Get project root
    project_root = Path(__file__).parent
    dashboard_dir = project_root / "src" / "dashboard"

    # Check if node_modules exists
    if not (dashboard_dir / "node_modules").exists():
        print("\n⚠️  node_modules not found. Installing dependencies...")
        print(f"   Running: npm install in {dashboard_dir}")

        try:
            subprocess.run(
                ["npm", "install"],
                cwd=str(dashboard_dir),
                check=True,
                shell=True  # Required for Windows
            )
            print("✅ Dependencies installed")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return
        except FileNotFoundError:
            print("❌ npm not found. Please install Node.js first.")
            print("   Download from: https://nodejs.org/")
            input("Press Enter to exit...")
            return

    # Start Flask backend
    print("\n🚀 Starting Flask backend (port 8000)...")
    backend_process = subprocess.Popen(
        [sys.executable, "app.py"],
        cwd=str(project_root)
    )

    # Give backend time to start
    print("   Waiting for backend to initialize...")
    time.sleep(3)

    # Start React frontend
    print("\n🚀 Starting React frontend (port 3000)...")
    print("   Dashboard will open at: http://localhost:3000")
    print("\n" + "=" * 70)
    print("✨ Dashboard is starting...")
    print("=" * 70)
    print("\nPress Ctrl+C to stop both servers\n")

    try:
        frontend_process = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=str(dashboard_dir),
            shell=True  # Needed on Windows so npm.cmd is resolved correctly
        )

        # Wait for processes
        frontend_process.wait()

    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
        frontend_process.terminate()
        backend_process.terminate()

        # Wait for graceful shutdown
        frontend_process.wait(timeout=5)
        backend_process.wait(timeout=5)

        print("✅ Dashboard stopped")

    except Exception as e:
        print(f"❌ Error: {e}")
        backend_process.terminate()
        if 'frontend_process' in locals():
            frontend_process.terminate()

if __name__ == "__main__":
    main()
