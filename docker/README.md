# Glass Docker Deployment

Complete Docker setup for Glass hallucination detection API.

---

## 📦 Quick Start

### Option 1: OpenAI Backend (Fast, Cloud)

```bash
# Copy environment file
cp docker/.env.example docker/.env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-your-key-here

# Start the service
cd docker
docker-compose --profile openai up -d

# Test
curl http://localhost:8000/health
```

### Option 2: Ollama Backend (Private, Local)

```bash
# Make sure Ollama is running on your host
ollama serve

# Start Glass with Ollama backend
cd docker
docker-compose --profile ollama up -d

# Test
curl http://localhost:8001/health
```

### Option 3: Full Stack (API + Ollama in Docker)

```bash
cd docker
docker-compose --profile ollama-docker --profile ollama up -d

# Wait for Ollama to start, then pull model
docker exec ollama ollama pull llama3.1:8b

# Test
curl http://localhost:8001/health
```

---

## 🔧 Configuration

### Environment Variables

Edit `docker/.env`:

```bash
# Backend selection
GLASS_BACKEND=openai  # or ollama

# OpenAI settings
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini

# Ollama settings
OLLAMA_HOST=http://host.docker.internal:11434
OLLAMA_MODEL=llama3.1:8b
OLLAMA_TIMEOUT=180.0

# Glass settings
GLASS_TEMPERATURE=0.3
GLASS_SYMMETRY_THRESHOLD=0.6
```

---

## 🚀 Usage Examples

### Evaluate Single Prompt

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": ["Who won the 2019 Nobel Prize in Physics?"],
    "h_star": 0.05
  }'
```

**Response:**
```json
{
  "results": [
    {
      "prompt": "Who won the 2019 Nobel Prize in Physics?",
      "decision": "answer",
      "symmetry_score": 0.850,
      "isr": 15.2,
      "roh_bound": 0.03
    }
  ],
  "total_time": 0.52,
  "average_time": 0.52,
  "backend": "gpt-4o-mini"
}
```

### Evaluate Multiple Prompts

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": [
      "What is the capital of France?",
      "Who wrote Romeo and Juliet?",
      "What is 2+2?"
    ],
    "h_star": 0.05,
    "symmetry_threshold": 0.7
  }'
```

### Health Check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "glass-api",
  "backend": "gpt-4o-mini",
  "timestamp": 1709234567.89
}
```

### API Info

```bash
curl http://localhost:8000/
```

**Response:**
```json
{
  "version": "1.0.0",
  "backend_type": "openai",
  "backend_model": "gpt-4o-mini",
  "glass_enabled": true,
  "features": [
    "grammatical_symmetry",
    "single_api_call",
    "30x_speedup",
    "ollama_support",
    "privacy_first"
  ]
}
```

---

## 🔍 Docker Services

### glass-api-openai
- **Port:** 8000
- **Backend:** OpenAI (GPT-4o-mini)
- **Profile:** `openai`
- **Use case:** Production, fast response times

### glass-api-ollama
- **Port:** 8001
- **Backend:** Ollama (llama3.1:8b)
- **Profile:** `ollama`
- **Use case:** Privacy-first, zero API costs

### ollama (optional)
- **Port:** 11434
- **Profile:** `ollama-docker`
- **Use case:** Run Ollama inside Docker

### nginx (optional)
- **Port:** 80
- **Profile:** `nginx`
- **Use case:** Load balancing, rate limiting

---

## 📊 Profiles

Docker Compose profiles allow you to run different configurations:

```bash
# OpenAI only
docker-compose --profile openai up

# Ollama only (requires Ollama on host)
docker-compose --profile ollama up

# Ollama + Ollama in Docker
docker-compose --profile ollama-docker --profile ollama up

# With Nginx load balancer
docker-compose --profile openai --profile nginx up

# Everything
docker-compose --profile openai --profile ollama --profile ollama-docker --profile nginx up
```

---

## 🛠️ Building

### Build Image

```bash
cd docker
docker build -t glass-api:latest -f Dockerfile ..
```

### Build with Compose

```bash
docker-compose build
```

### Multi-platform Build

```bash
docker buildx build --platform linux/amd64,linux/arm64 -t glass-api:latest ..
```

---

## 📈 Scaling

### Horizontal Scaling (Multiple Instances)

Edit `docker-compose.yml`:

```yaml
services:
  glass-api-openai:
    deploy:
      replicas: 3  # Run 3 instances
```

Then enable Nginx profile for load balancing:

```bash
docker-compose --profile openai --profile nginx up --scale glass-api-openai=3
```

### Vertical Scaling (More Resources)

Edit `docker-compose.yml`:

```yaml
services:
  glass-api-openai:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '1.0'
          memory: 1G
```

---

## 🔒 Security

### Production Checklist

- [ ] Use secrets for API keys (not environment variables)
- [ ] Enable HTTPS (add SSL certificates to Nginx)
- [ ] Restrict CORS origins in nginx.conf
- [ ] Set up rate limiting (already configured in Nginx)
- [ ] Use firewall rules to restrict access
- [ ] Enable Docker security scanning
- [ ] Run containers as non-root user

### Using Docker Secrets

Create `docker-compose.prod.yml`:

```yaml
services:
  glass-api-openai:
    secrets:
      - openai_api_key
    environment:
      - OPENAI_API_KEY_FILE=/run/secrets/openai_api_key

secrets:
  openai_api_key:
    external: true
```

---

## 📊 Monitoring

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f glass-api-openai

# Last 100 lines
docker-compose logs --tail=100 glass-api-openai
```

### Container Stats

```bash
docker stats
```

### Health Check

```bash
docker inspect --format='{{.State.Health.Status}}' glass-api-openai
```

---

## 🧪 Testing

### Test OpenAI Backend

```bash
docker-compose --profile openai up -d
sleep 5  # Wait for startup

# Test health
curl http://localhost:8000/health

# Test evaluation
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"prompts": ["What is 2+2?"]}'
```

### Test Ollama Backend

```bash
# Make sure Ollama is running
ollama serve &

# Start Glass
docker-compose --profile ollama up -d
sleep 5

# Test (slower due to local inference)
curl -X POST http://localhost:8001/evaluate \
  -H "Content-Type: application/json" \
  -d '{"prompts": ["What is 2+2?"]}' \
  --max-time 200
```

---

## 🐛 Troubleshooting

### Issue: Container exits immediately

**Solution:** Check logs and environment variables

```bash
docker-compose logs glass-api-openai
docker-compose config  # Validate compose file
```

### Issue: Cannot connect to Ollama from Docker

**Solution:** Use `host.docker.internal` instead of `localhost`

```bash
# In .env
OLLAMA_HOST=http://host.docker.internal:11434
```

### Issue: Timeout errors with Ollama

**Solution:** Increase timeout in .env

```bash
OLLAMA_TIMEOUT=300.0  # 5 minutes
```

### Issue: Health check failing

**Solution:** Wait longer for startup or check backend connectivity

```bash
docker-compose ps
docker-compose logs
```

---

## 🔄 Updates

### Update Glass Code

```bash
# Pull latest code
git pull

# Rebuild and restart
docker-compose down
docker-compose build
docker-compose --profile openai up -d
```

### Update Dependencies

Edit `Dockerfile` to update package versions, then rebuild.

---

## 🧹 Cleanup

### Stop Services

```bash
docker-compose down
```

### Remove Volumes

```bash
docker-compose down -v
```

### Remove Images

```bash
docker rmi glass-api:latest
```

### Full Cleanup

```bash
docker-compose down -v --rmi all
docker system prune -a
```

---

## 📚 Additional Resources

- **API Documentation:** See `glass/api.py`
- **Deployment Guide:** `DEPLOYMENT_GUIDE.md`
- **Glass Documentation:** `glass/README_EN.md`
- **Nginx Configuration:** `docker/nginx.conf`

---

**Glass is production-ready! Deploy with confidence using Docker.** 🚀🐳
