import os
import threading
import time
from flask import Flask, jsonify, send_from_directory

app = Flask(__name__)
port = int(os.environ.get('PORT', 5000))

_WEBAPP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webapp")
_real_app = None

@app.route('/')
def index():
    if _real_app:
        return send_from_directory(
            os.path.join(_WEBAPP_DIR, "templates"), "index.html")
    return """
    <html>
    <head>
        <title>EmotiSense Loading...</title>
        <meta http-equiv="refresh" content="10">
        <style>
            body { font-family: Arial; display: flex; justify-content: center; 
                   align-items: center; height: 100vh; margin: 0; background: #0f172a; color: white; }
            .box { text-align: center; }
            h1 { color: #06b6d4; }
            p { color: #94a3b8; }
            .spinner { font-size: 50px; animation: spin 2s linear infinite; display: inline-block; }
            @keyframes spin { 0%{transform:rotate(0deg)} 100%{transform:rotate(360deg)} }
        </style>
    </head>
    <body>
        <div class="box">
            <div class="spinner">⚙️</div>
            <h1>EmotiSense is starting up...</h1>
            <p>Loading AI models, please wait. Page refreshes automatically every 10 seconds.</p>
        </div>
    </body>
    </html>
    """, 200

@app.route('/api/status')
def status():
    return jsonify({"status": "ready" if _real_app else "loading"}), 200

def background_load():
    global _real_app
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, _WEBAPP_DIR)
    time.sleep(2)
    try:
        from webapp.app import app as real, load_model
        CHECKPOINT = os.path.join("experiments", "checkpoints", "best_model.pt")
        load_model(CHECKPOINT if os.path.exists(CHECKPOINT) else None)
        # Register all real routes
        for rule in real.url_map._rules:
            if rule.endpoint == 'static':
                continue
            try:
                app.add_url_rule(
                    rule.rule,
                    endpoint=rule.endpoint,
                    view_func=real.view_functions[rule.endpoint],
                    methods=list(rule.methods)
                )
            except Exception:
                pass
        _real_app = real
        print("[Render] Full app ready!")
    except Exception as e:
        print(f"[Render] Load error: {e}")

threading.Thread(target=background_load, daemon=True).start()
print(f"[Render] Quick-start Flask on port {port}")
app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False, threaded=True)
