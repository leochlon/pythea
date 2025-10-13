"""
Glass REST API
==============

Production-ready FastAPI server for Glass hallucination detection.

Usage:
    uvicorn glass.api:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, validator
from typing import List, Optional
import time
import os
import logging

try:
    from glass import GlassPlanner, GlassItem
    from hallbayes import OpenAIBackend
    from hallbayes.htk_backends import OllamaBackend
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from glass import GlassPlanner, GlassItem
    from hallbayes import OpenAIBackend
    from hallbayes.htk_backends import OllamaBackend

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('glass.api')

# Initialize FastAPI
app = FastAPI(
    title="Glass API",
    description="30× faster hallucination detection using grammatical symmetry",
    version="1.0.0"
)

# Initialize backend based on environment
BACKEND_TYPE = os.environ.get('GLASS_BACKEND', 'openai')
logger.info(f"Initializing backend: {BACKEND_TYPE}")

if BACKEND_TYPE == 'ollama':
    backend = OllamaBackend(
        model=os.environ.get('OLLAMA_MODEL', 'llama3.1:8b'),
        host=os.environ.get('OLLAMA_HOST', 'http://localhost:11434'),
        request_timeout=float(os.environ.get('OLLAMA_TIMEOUT', '180.0'))
    )
    logger.info(f"Ollama backend initialized: {backend.model}")
else:
    backend = OpenAIBackend(
        model=os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')
    )
    logger.info(f"OpenAI backend initialized: {backend.model}")

# Initialize planner
planner = GlassPlanner(
    backend,
    temperature=float(os.environ.get('GLASS_TEMPERATURE', '0.3'))
)
logger.info("Glass planner initialized")


# Request/Response Models
class EvaluateRequest(BaseModel):
    """Request model for evaluation endpoint"""
    prompts: List[str]
    h_star: float = 0.05
    symmetry_threshold: Optional[float] = None

    @validator('prompts')
    def validate_prompts(cls, v):
        if not v:
            raise ValueError("prompts cannot be empty")
        if len(v) > 100:
            raise ValueError("maximum 100 prompts per request")
        for prompt in v:
            if not prompt.strip():
                raise ValueError("prompts cannot contain empty strings")
            if len(prompt) > 10000:
                raise ValueError("prompt too long (max 10000 characters)")
        return v

    @validator('h_star')
    def validate_h_star(cls, v):
        if not 0 < v < 1:
            raise ValueError("h_star must be between 0 and 1")
        return v


class EvaluationResult(BaseModel):
    """Single evaluation result"""
    prompt: str
    decision: str
    symmetry_score: float
    isr: float
    roh_bound: float
    response_text: Optional[str] = None


class EvaluateResponse(BaseModel):
    """Response model for evaluation endpoint"""
    results: List[EvaluationResult]
    total_time: float
    average_time: float
    backend: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    backend: str
    timestamp: float


class InfoResponse(BaseModel):
    """API information response"""
    version: str
    backend_type: str
    backend_model: str
    glass_enabled: bool
    features: List[str]


# Endpoints
@app.get("/", response_model=InfoResponse)
async def root():
    """Root endpoint with API information"""
    return InfoResponse(
        version="1.0.0",
        backend_type=BACKEND_TYPE,
        backend_model=backend.model,
        glass_enabled=True,
        features=[
            "grammatical_symmetry",
            "single_api_call",
            "30x_speedup",
            "ollama_support",
            "privacy_first"
        ]
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        service="glass-api",
        backend=backend.model,
        timestamp=time.time()
    )


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(request: EvaluateRequest):
    """
    Evaluate prompts for hallucination risk using Glass.

    Returns decision (ANSWER/REFUSE), symmetry score, and EDFL metrics.
    """
    try:
        logger.info(f"Received evaluation request for {len(request.prompts)} prompts")
        start = time.time()

        # Create items
        items = []
        for prompt in request.prompts:
            item = GlassItem(prompt=prompt)
            if request.symmetry_threshold is not None:
                item.symmetry_threshold = request.symmetry_threshold
            items.append(item)

        # Run evaluation
        metrics = planner.run(items, h_star=request.h_star)
        elapsed = time.time() - start

        # Format results
        results = []
        for i, m in enumerate(metrics):
            result = EvaluationResult(
                prompt=request.prompts[i],
                decision="answer" if m.decision_answer else "refuse",
                symmetry_score=m.symmetry_score if hasattr(m, 'symmetry_score') else 0.0,
                isr=m.isr,
                roh_bound=m.roh_bound,
                response_text=m.response_text if hasattr(m, 'response_text') else None
            )
            results.append(result)

        logger.info(
            f"Evaluation completed: {len(request.prompts)} prompts, "
            f"{sum(1 for r in results if r.decision == 'answer')} answered, "
            f"{elapsed:.2f}s total"
        )

        return EvaluateResponse(
            results=results,
            total_time=elapsed,
            average_time=elapsed / len(request.prompts),
            backend=backend.model
        )

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    """
    Prometheus-compatible metrics endpoint.

    Note: This is a placeholder. Use MonitoredPlanner for full metrics.
    """
    return {
        "status": "metrics_placeholder",
        "note": "Use MonitoredPlanner wrapper for full Prometheus metrics"
    }


# Run with: uvicorn glass.api:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
