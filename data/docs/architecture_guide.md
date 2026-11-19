# Distributed Architecture & Microservices Guidelines

## 1. Overview
This guide describes the microservices architecture, API gateway patterns, distributed caching, and vector indexing standards for our production platform.

## 2. API Gateway Routing & Security
All incoming external requests terminate at the API Gateway.
- Authentication: JWT tokens with RS256 signature verification.
- Rate Limiting: Token bucket algorithm configured to 100 requests per second per client IP.
- Circuit Breaker: Automatically opens when error rates exceed 50% over a 10-second rolling window.

## 3. Distributed Caching with Redis
To minimize database load, read-heavy queries utilize a multi-tier caching strategy:
- L1 Cache: In-memory LRU cache inside the application process (TTL: 60 seconds).
- L2 Cache: Distributed Redis cluster with replication (TTL: 15 minutes).
- Cache Invalidation: Write-through cache with Redis Pub/Sub events broadcasting cache purges across worker nodes.

## 4. Vector Database & Semantic Retrieval
Semantic search workflows employ ChromaDB for high-throughput vector similarity calculations.
- Metric: Cosine distance is used for normalized dense embeddings.
- Chunking: Technical documents are segmented into 400-word chunks with a 50-word sliding overlap window.
- Confidence Threshold: Responses must meet a minimum similarity confidence score of 0.60; queries falling below this threshold are refused.

## 5. Fault Tolerance & Failover
- Health Checks: Services expose `/health` endpoints polled every 15 seconds.
- Graceful Shutdown: Worker processes trap SIGTERM, finish in-flight requests within a 30-second deadline, and cleanly close database connection pools.
