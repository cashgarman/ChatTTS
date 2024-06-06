#!/bin/bash

# Navigate to the directory containing the index.html file
cd "$(dirname "$0")"

# Start a simple HTTP server on port 8000
python3 -m http.server 8000