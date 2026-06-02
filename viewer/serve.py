from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import os


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    os.chdir(ROOT)
    address = ("127.0.0.1", 8000)
    print(f"Serving {ROOT} at http://{address[0]}:{address[1]}/viewer/")
    print("Press Ctrl+C to stop.")
    ThreadingHTTPServer(address, SimpleHTTPRequestHandler).serve_forever()


if __name__ == "__main__":
    main()