#!/bin/sh
# Start front-end server
cd frontend
python -m http.server 8000 &
echo "Front-end started on http://localhost:8000"

# Start API server
cd ../backend
python api.py &
echo "API started on port 5000"

# Keep the container running
wait
