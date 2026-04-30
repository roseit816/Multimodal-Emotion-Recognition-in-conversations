import os, sys, threading, time, traceback
from flask import Flask, Response, send_from_directory, jsonify

PORT = int(os.environ.get('PORT', 5000))
ROOT = os.path.dirname(os.path.abspath(__file__))
WEBAPP = os.path.join(ROOT, 'webapp')
TMPL = os.path.join(WEBAPP, 'templates')
STAT = os.path.join(WEBAPP, 'static')

sys.path.insert(0, ROOT)
sys.path.insert(0, WEBAPP)

app = Flask(__name__, template_folder=TMPL, static_folder=STAT)
_ready = False
_error = None

@app.route('/')
def index():
    if _error:
        return f"<pre style='color:red;background:#111;padding:20px'>{_error}</pre>", 500
    if not _ready:
        return """<html><head><meta http-equiv='refresh' content='10'>
        <style>body{background:#0b0d17;color:#dde1ff;font-family:sans-serif;
        display:flex;align-items:center;justify-content:center;height:100vh;margin:0}
        .b{text-align:center}.e{font-size:60px}.h{color:#5865f2;font-size:24px;margin:10px 0}
        .p{color:#6872a8}</style></head>
        <body><div class='b'><div class='e'>🧠</div>
        <div class='h'>EmotiSense is starting up...</div>
        <div class='p'>Loading AI models. Page auto-refreshes every 10 seconds.</div>
        </div></body></html>""", 200
    return send_from_directory(TMPL, 'index.html')

@app.route('/analytics')
def analytics():
    if not _ready:
        return "<p>Still loading...</p>", 200
    return send_from_directory(TMPL, 'analytics.html')

@app.route('/static/<path:f>')
def static_f(f):
    return send_from_directory(STAT, f)

@app.route('/api/status')
def status():
    if not _ready:
        return jsonify({"status": "loading", "model_loaded": False}), 200
    from webapp.app import app as real
    return real.view_functions['status']()

def make_proxy(fn_name):
    def proxy(**kwargs):
        from webapp.app import app as real
        return real.view_functions[fn_name](**kwargs)
    proxy.__name__ = fn_name
    return proxy

def load_background():
    global _ready, _error
    time.sleep(2)
    try:
        print("[Render] Importing webapp...")
        from webapp.app import app as real, load_model
        print("[Render] Imported! Loading model...")
        ckpt = os.path.join(ROOT, 'experiments', 'checkpoints', 'best_model.pt')
        load_model(ckpt if os.path.exists(ckpt) else None)
        print("[Render] Model loaded! Registering routes...")
        for rule in real.url_map._rules:
            ep = rule.endpoint
            if ep in ('static', 'index', 'analytics', 'status'):
                continue
            try:
                app.add_url_rule(
                    rule.rule, endpoint=ep,
                    view_func=make_proxy(ep),
                    methods=list(rule.methods)
                )
            except Exception:
                pass
        _ready = True
        print("[Render] ✅ App fully ready!")
    except Exception:
        _error = traceback.format_exc()
        print(f"[Render] ❌ FAILED:\n{_error}")

threading.Thread(target=load_background, daemon=True).start()
print(f"[Render] Starting on port {PORT}")
app.run(host='0.0.0.0', port=PORT, debug=False, use_reloader=False, threaded=True)
