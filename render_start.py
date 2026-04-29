import os, sys, threading

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_WEBAPP_DIR   = os.path.join(_PROJECT_ROOT, "webapp")
for _p in [_PROJECT_ROOT, _WEBAPP_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

port = int(os.environ.get('PORT', 5000))

# Import the REAL app directly — it already has all routes + templates
from webapp.app import app, load_model

def load_in_background():
    import time
    time.sleep(1)
    CHECKPOINT = os.path.join("experiments", "checkpoints", "best_model.pt")
    load_model(CHECKPOINT if os.path.exists(CHECKPOINT) else None)
    print("[Render] Model loaded!")

threading.Thread(target=load_in_background, daemon=True).start()

print(f"[Render] Starting on port {port}")
app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
