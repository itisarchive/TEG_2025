#!/bin/bash
set -euo pipefail

cat <<'HEADER'
💾 Ending GraphRAG Session
===========================
HEADER

if ! docker ps | grep -q neo4j; then
    echo "ℹ️  Neo4j is not running — nothing to stop."
    exit 0
fi

running_count=$(docker ps | grep -c neo4j || true)
echo "📊 Neo4j containers running: ${running_count}"

echo "⏹️  Stopping Neo4j..."
docker-compose down

cat <<'DONE'

✅ Session ended successfully!

📋 What happened:
  • Neo4j stopped cleanly
  • All data is preserved in Docker volumes
  • Database will be exactly as you left it

🚀 To continue working:
  • Run: ./start_session.sh
  • Or:  docker-compose up -d

💡 Your data persists automatically — no manual saving needed!
DONE
