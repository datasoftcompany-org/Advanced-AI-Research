"""Serve the sentiment browser interface on the loopback address."""

import argparse
import csv
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import secrets
import threading

import classify

WEB = Path(__file__).resolve().parent / "web"
MAX_REQUEST = 10 * 1024 * 1024


def parse_input(data):
    if not isinstance(data, dict) or data.get("model") not in classify.MODELS:
        raise ValueError("Choose a supported local model.")
    if data.get("mode") == "text":
        text = data.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Enter a non-empty English review.")
        return [text]
    if data.get("mode") == "csv":
        content, column = data.get("csv"), data.get("column")
        if not isinstance(content, str) or not isinstance(column, str) or not column:
            raise ValueError("Select a CSV file and enter its review column.")
        return classify.read_csv(io.StringIO(content.lstrip("\ufeff"), newline=""), column)
    raise ValueError("Choose text or CSV input.")


class Handler(BaseHTTPRequestHandler):
    def send_bytes(self, status, content, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; script-src 'self'; style-src 'self'; "
                         "connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'")
        self.end_headers()
        self.wfile.write(content)

    def valid_host(self):
        return self.headers.get("Host") == self.server.authority

    def do_GET(self):
        if not self.valid_host():
            self.send_bytes(403, b"Invalid host", "text/plain")
            return
        routes = {
            "/": (WEB / "index.html", "text/html; charset=utf-8"),
            "/app.js": (WEB / "app.js", "text/javascript; charset=utf-8"),
            "/style.css": (WEB / "style.css", "text/css; charset=utf-8"),
            "/glass.css": (classify.ROOT / "assets/glass.css", "text/css; charset=utf-8"),
        }
        if self.path not in routes:
            self.send_bytes(404, b"Not found", "text/plain")
            return
        path, content_type = routes[self.path]
        content = path.read_bytes()
        if self.path == "/":
            content = content.replace(b"__TOKEN__", self.server.token.encode())
        self.send_bytes(200, content, content_type)

    def do_POST(self):
        if (not self.valid_host()
                or self.headers.get("X-Session-Token") != self.server.token
                or self.headers.get("Origin") not in (None, "http://" + self.server.authority)):
            self.send_bytes(403, b"Invalid session", "text/plain")
            return
        if self.path != "/api/classify":
            self.send_bytes(404, b"Not found", "text/plain")
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= MAX_REQUEST:
                raise ValueError("Request must be between 1 byte and 10 MiB.")
            data = json.loads(self.rfile.read(size))
            texts = parse_input(data)
        except (ValueError, UnicodeError, csv.Error) as error:
            self.send_bytes(400, str(error).encode(), "text/plain; charset=utf-8")
            return
        if not self.server.inference_lock.acquire(blocking=False):
            self.send_bytes(409, b"Another classification is running. Please wait.", "text/plain")
            return
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

            def event(value):
                self.wfile.write((json.dumps(value) + "\n").encode())
                self.wfile.flush()

            try:
                settings = json.loads((classify.ROOT / "research/local-sentiment-evaluation/protocol.json").read_text())
                settings["model"] = data["model"]
                event({"type": "progress", "completed": 0, "total": len(texts)})
                rows = classify.classify(
                    texts, settings,
                    lambda done, total: event({"type": "progress", "completed": done, "total": total}),
                )
                event({"type": "done", "rows": rows, "csv": classify.csv_text(rows)})
            except (OSError, ValueError, KeyError) as error:
                event({"type": "error", "message": str(error)})
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.server.inference_lock.release()

    def log_message(self, format, *args):
        # Review text and uploaded file contents are never logged.
        pass


def create_server(port):
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.authority = f"127.0.0.1:{server.server_port}"
    server.token = secrets.token_urlsafe(32)
    server.inference_lock = threading.Lock()
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = create_server(args.port)
    print(f"Open http://{server.authority} in your browser. Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
