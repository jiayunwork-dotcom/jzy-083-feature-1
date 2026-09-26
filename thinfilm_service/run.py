"""Service entry point: serve the Flask app with waitress (multi-threaded).

Waitress handles concurrent requests on a thread pool; every request builds
its own stack and matrix intermediates, so concurrent computations are fully
independent.
"""

from __future__ import annotations

import os

from waitress import serve

from thinopt import create_app

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8080"))
THREADS = int(os.environ.get("THREADS", "8"))

if __name__ == "__main__":
    app = create_app()
    serve(app, host=HOST, port=PORT, threads=THREADS)
