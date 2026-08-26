#!/usr/bin/env python3
"""
Simple HTTP server to view the training montage HTML locally.
Serves videos from the training directory with proper CORS headers.

Run:
  python serve_montage.py
Then open:
  http://localhost:8000/training_montage.html
"""

import http.server
import socketserver
import os
from pathlib import Path
import webbrowser
import threading

PORT = 8000
TRAINING_DIR = Path(__file__).parent

class VideoHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(TRAINING_DIR), **kwargs)

    def end_headers(self):
        # Add CORS headers for video serving
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Accept-Ranges', 'bytes')
        super().end_headers()

    def log_message(self, format, *args):
        # Quiet logging
        if '200 --' not in format % args:
            print(f"[{self.client_address[0]}] {format % args}")

def main():
    os.chdir(TRAINING_DIR)

    with socketserver.TCPServer(("", PORT), VideoHandler) as httpd:
        url = f"http://localhost:{PORT}/training_montage.html"
        print(f"\n{'='*60}")
        print(f"  Training Montage Server")
        print(f"{'='*60}")
        print(f"\n  📹 Serving: {TRAINING_DIR}")
        print(f"  🌐 Open:   {url}\n")
        print(f"  Press Ctrl+C to stop\n")
        print(f"{'='*60}\n")

        # Try to open browser automatically
        try:
            webbrowser.open(url)
            print(f"  ✓ Opening browser...\n")
        except:
            pass

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n  Server stopped.")

if __name__ == "__main__":
    main()
