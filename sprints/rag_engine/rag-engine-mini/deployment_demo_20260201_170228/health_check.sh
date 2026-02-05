#!/bin/bash
# Health check script for all services

echo "🔍 Checking RAG Engine Health..."
echo ""

# Check API
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ API: Healthy"
    API_STATUS=$(curl -s http://localhost:8000/health)
    echo "   Response: $API_STATUS"
else
    echo "❌ API: Not responding"
fi

# Check PostgreSQL
if docker exec rag-engine-postgres pg_isready -U rag_admin > /dev/null 2>&1; then
    echo "✅ PostgreSQL: Running"
else
    echo "❌ PostgreSQL: Not running"
fi

# Check Redis
if docker exec rag-engine-redis redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis: Running"
else
    echo "❌ Redis: Not running"
fi

# Check Qdrant
if curl -s http://localhost:6333/healthz > /dev/null 2>&1; then
    echo "✅ Qdrant: Running"
else
    echo "❌ Qdrant: Not running"
fi

# Check Nginx
if curl -s http://localhost > /dev/null 2>&1; then
    echo "✅ Nginx: Running"
else
    echo "❌ Nginx: Not running"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
