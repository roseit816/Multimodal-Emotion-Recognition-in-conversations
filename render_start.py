import os, sys

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_WEBAPP_DIR   = os.path.join(_PROJECT_ROOT, "webapp")
for _p in [_PROJECT_ROOT, _WEBAPP_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from webapp.app import app, load_model

CHECKPOINT = os.path.join("experiments", "checkpoints", "best_model.pt")
load_model(CHECKPOINT if os.path.exists(CHECKPOINT) else None)

port = int(os.environ.get('PORT', 5000))
print(f"[Render] Starting on port {port}")

app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
