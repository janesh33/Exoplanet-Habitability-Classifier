"""
Root App Launcher
=================
Forwards requests to 06_WEBSITE/backend/app.py for seamless execution.
"""

import os
import sys
import importlib.util

backend_app_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '06_WEBSITE', 'backend', 'app.py'
)

if not os.path.exists(backend_app_path):
    raise FileNotFoundError(f"Backend app not found at {backend_app_path}")

spec = importlib.util.spec_from_file_location("backend_app", backend_app_path)
backend_app = importlib.util.module_from_spec(spec)
sys.modules["backend_app"] = backend_app
spec.loader.exec_module(backend_app)

app = backend_app.app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
