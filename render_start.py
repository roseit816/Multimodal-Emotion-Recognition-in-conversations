import os, sys, threading

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_WEBAPP_DIR   = os.path.join(_PROJECT_ROOT, "webapp")
for _p in [_PROJECT_ROOT, _WEBAPP_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from flask import Flask, jsonify
from flask_cors import CORS

# Start a tiny Flask app IMMEDIATELY on the correct port
bootstrap = Flask(__name__)
CORS(bootstrap)
port = int(os.environ.get('PORT', 5000))

@bootstrap.route('/')
def loading():
    return "<h1>EmotiSense is loading... please wait 60 seconds and refresh.</h1>"

@bootstrap.route('/api/status')
def status():
    return jsonify({"status": "loading"})

def load_full_app():
    """Load the real app in background after Flask is already running."""
    import time
    time.sleep(3)  # Let Flask bind first
    
    from webapp.app import app as real_app, load_model
    CHECKPOINT = os.path.join("experiments", "checkpoints", "best_model.pt")
    load_model(CHECKPOINT if os.path.exists(CHECKPOINT) else None)
    
    # Copy all routes from real app to bootstrap app
    for rule in real_app.url_map._rules:
        if rule.endpoint == 'static':
            continue
        try:
            bootstrap.add_url_rule(
                rule.rule,
                endpoint=rule.endpoint + '_real',
                view_func=real_app.view_functions[rule.endpoint],
                methods=rule.methods
            )
        except Exception:
            pass
    print("[Render] Full app loaded and routes registered!")

# Load real app in background thread
threading.Thread(target=load_full_app, daemon=True).start()

print(f"[Render] Bootstrap server starting on port {port}")
bootstrap.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
