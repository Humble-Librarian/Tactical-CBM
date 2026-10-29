"""
Tactical CBM Engine - Dashboard Server (Stage 5)
Serves the interactive tactical sandbox on local HTTP port.
"""
import sys
import os
import http.server
import socketserver
import webbrowser
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DASHBOARD_DIR = BASE_DIR / "dashboard"

from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

PORT = 8501

class CustomHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)

    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

def start_server(open_browser: bool = False):
    print("=" * 65)
    print("STAGE 5: TACTICAL-CBM INTERACTIVE DASHBOARD")
    print("=" * 65)
    print(f"Serving dashboard from: {DASHBOARD_DIR}")
    print(f"URL: http://127.0.0.1:{PORT}")
    print("Press Ctrl+C to stop the server.")
    print("=" * 65)
    
    if open_browser:
        webbrowser.open(f"http://127.0.0.1:{PORT}")
        
    server_address = ("0.0.0.0", PORT)
    ThreadingHTTPServer.allow_reuse_address = True
    with ThreadingHTTPServer(server_address, CustomHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8501, help="Port to serve on (default: 8501)")
    parser.add_argument("--open", action="store_true", help="Automatically open browser")
    args = parser.parse_args()
    PORT = args.port
    start_server(open_browser=args.open)
