#!/bin/bash
PORT=8502

echo ""
echo "  alpha-bench"
echo "  Open in browser: http://localhost:$PORT"
echo ""

streamlit run app.py --server.port $PORT
