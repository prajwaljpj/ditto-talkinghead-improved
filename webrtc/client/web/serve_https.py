#!/usr/bin/env python3
"""
Simple HTTPS server for local development.
Generates a self-signed certificate if one doesn't exist.
"""

import http.server
import ssl
import os
from pathlib import Path

# Certificate files
CERT_FILE = "localhost.pem"
KEY_FILE = "localhost-key.pem"

def generate_certificate():
    """Generate self-signed certificate for localhost."""
    print("Generating self-signed certificate...")
    os.system(f"""
openssl req -x509 -newkey rsa:4096 -nodes \
    -keyout {KEY_FILE} \
    -out {CERT_FILE} \
    -days 365 \
    -subj "/CN=localhost" \
    -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"
    """)
    print(f"Certificate generated: {CERT_FILE}, {KEY_FILE}")

def main():
    port = 8000

    # Generate certificate if it doesn't exist
    if not os.path.exists(CERT_FILE) or not os.path.exists(KEY_FILE):
        generate_certificate()

    # Create HTTPS server
    server_address = ('0.0.0.0', port)
    httpd = http.server.HTTPServer(server_address, http.server.SimpleHTTPRequestHandler)

    # Wrap with SSL
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(CERT_FILE, KEY_FILE)
    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)

    print(f"Serving HTTPS on 0.0.0.0 port {port}")
    print(f"Access at: https://localhost:{port}")
    print("\nNote: You'll see a security warning (self-signed cert).")
    print("Click 'Advanced' -> 'Proceed to localhost' to continue.\n")

    httpd.serve_forever()

if __name__ == "__main__":
    main()
