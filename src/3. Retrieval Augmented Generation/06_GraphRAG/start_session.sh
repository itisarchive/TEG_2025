#!/bin/bash
set -euo pipefail

cat <<'HEADER'
🚀 Starting GraphRAG Session
=============================
HEADER

if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop and try again."
    exit 1
fi

echo "▶️  Starting Neo4j..."
docker-compose up -d

echo "⏳ Waiting for Neo4j to be ready..."
sleep 10

echo "🔍 Checking system status..."
uv run python 0_setup.py --check

cat <<'READY'

✅ GraphRAG session started!

🎯 What you can do now:
  • Query the graph:      uv run python 3_query_knowledge_graph.py
  • Open Neo4j Browser:   http://localhost:7474
  • Check status anytime: uv run python 0_setup.py --check
  • End session:          ./end_session.sh

🔑 Neo4j Browser credentials:
  Username: neo4j
  Password: password123
READY
