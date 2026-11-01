#!/usr/bin/env python3
"""Local web page where the human approves what the robot may throw away.

tidy_desk.py writes <dir>/request.json (items with a crop image each) and waits
for <dir>/response.json. This page shows the request in Japanese with three
buttons per item: 捨てる / 保管箱へ / そのまま. Nothing is discarded without a
click on 捨てる.

    python3 confirm_server.py DIR [--host 127.0.0.1] [--port 8765]
    (--host 0.0.0.0 lets a phone on the same LAN open it)
"""
import argparse
import html
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

CHOICES = {"discard": "捨てる", "keep": "保管箱へ", "leave": "そのまま"}

PAGE = """<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>片付けの確認</title>
<style>
:root{{--bg:#f6f5f2;--card:#fff;--ink:#222;--muted:#666;--line:#ddd;--red:#c62828;--green:#2e7d32}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1b1b1b;--card:#262626;--ink:#eee;--muted:#aaa;--line:#444}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,sans-serif}}
main{{max-width:720px;margin:0 auto;padding:16px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px;margin:12px 0;
display:flex;gap:12px;flex-wrap:wrap}}
.card img{{width:160px;height:160px;object-fit:contain;background:#000;border-radius:8px}}
.info{{flex:1;min-width:200px}} .muted{{color:var(--muted);font-size:14px}}
button{{font-size:16px;padding:10px 14px;margin:4px 4px 0 0;border-radius:8px;border:1px solid var(--line);
background:var(--card);color:var(--ink);cursor:pointer}}
button.discard{{border-color:var(--red);color:var(--red)}} button.keep{{border-color:var(--green);color:var(--green)}}
button.suggested{{font-weight:700;box-shadow:0 0 0 2px currentColor inset}}
.done{{font-weight:700}}
</style></head><body><main>
<h1>片付けの確認</h1><p><a href="/notes">知識ノートを見る →</a></p>
{body}
</main>
<script>setTimeout(()=>location.reload(),15000)</script>
</body></html>"""


def render(folder):
    req_path, res_path = folder / "request.json", folder / "response.json"
    if not req_path.exists():
        return "<p class='muted'>いまは確認待ちの物はありません。</p>"
    req = json.loads(req_path.read_text())
    answered = json.loads(res_path.read_text())["decisions"] if res_path.exists() else {}
    cards = []
    for item in req["items"]:
        key = html.escape(item["key"])
        chosen = answered.get(item["key"])
        buttons = "".join(
            f"<button name='d' value='{c}' class='{c}{' suggested' if c == (chosen or item['suggest']) else ''}'>"
            f"{label}</button>" for c, label in CHOICES.items())
        state = f"<p class='done'>→ {CHOICES[chosen]}（変更できます）</p>" if chosen else ""
        action = (f"{state}<form method='post' action='/decide'><input type='hidden' name='key' value='{key}'>"
                  f"{buttons}</form>")
        if item.get("result"):
            action = f"<p class='done'>✓ {html.escape(item['result'])}</p>"
        elif item.get("info"):
            action = f"<p class='muted'>{html.escape(item['info'])}</p>"
        cards.append(
            f"<div class='card'><img src='/img/{html.escape(item['image'])}' alt=''>"
            f"<div class='info'><b>{html.escape(item['name'])}</b>"
            f"<div class='muted'>判定: {html.escape(item['category_ja'])}（確信 {item['confidence']:.0%}）</div>"
            f"<div>{html.escape(item['reason'])}</div>{action}</div></div>")
    head = f"<p class='muted'>ラウンド {req['round']}：机の上の物 {len(req['items'])} 件。</p>"
    return head + "".join(cards)


def make_handler(folder, notes=None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, body, ctype="text/html; charset=utf-8"):
            data = body if isinstance(body, bytes) else body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.startswith("/notes/img/") and notes is not None:
                path = notes.root / "crops" / Path(self.path[11:]).name
                if path.suffix == ".jpg" and path.exists():
                    return self._send(200, path.read_bytes(), "image/jpeg")
                return self._send(404, "not found")
            if self.path.startswith("/notes") and notes is not None:
                import desk_memory
                notes.reload() if hasattr(notes, "reload") else None
                body = "<h1>知識ノート</h1><p><a href='/'>← 確認ページへ</a></p>" + desk_memory.notes_html(notes)
                return self._send(200, PAGE.format(body=body))
            if self.path.startswith("/img/"):
                path = (folder / Path(self.path[5:]).name)
                if path.suffix == ".jpg" and path.exists():
                    return self._send(200, path.read_bytes(), "image/jpeg")
                return self._send(404, "not found")
            self._send(200, PAGE.format(body=render(folder)))

        def do_POST(self):
            form = parse_qs(self.rfile.read(int(self.headers.get("Content-Length", 0))).decode())
            key, choice = form.get("key", [""])[0], form.get("d", [""])[0]
            req_path, res_path = folder / "request.json", folder / "response.json"
            if choice in CHOICES and req_path.exists():
                keys = {i["key"] for i in json.loads(req_path.read_text())["items"]}
                if key in keys:
                    decisions = json.loads(res_path.read_text())["decisions"] if res_path.exists() else {}
                    decisions[key] = choice
                    res_path.write_text(json.dumps({"decisions": decisions}, ensure_ascii=False))
            self.send_response(303)
            self.send_header("Location", "/")
            self.end_headers()

    return Handler


def serve(folder, host="127.0.0.1", port=8765, notes=None):
    """Start the page in a daemon thread; returns the server (call .shutdown() to stop).

    notes: a desk_memory.Notebook shown at /notes (what the robot has learned so far)."""
    server = ThreadingHTTPServer((host, port), make_handler(Path(folder), notes))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dir")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    a = p.parse_args()
    Path(a.dir).mkdir(parents=True, exist_ok=True)
    print(f"http://{a.host}:{a.port}/", flush=True)
    ThreadingHTTPServer((a.host, a.port), make_handler(Path(a.dir))).serve_forever()
