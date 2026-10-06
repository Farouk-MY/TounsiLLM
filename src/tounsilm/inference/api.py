"""
FastAPI Local Inference Server for TounsiLM.

Provides endpoints:
  - GET  /health      : Healthcheck and device status
  - GET  /model-info  : Architecture details and parameter counts
  - POST /generate    : Autoregressive text generation
"""

from __future__ import annotations

import os
from typing import Optional

from pydantic import BaseModel, Field

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from tounsilm.inference.generator import TextGenerator


class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt in Tunisian Arabic, Arabizi, or mixed")
    max_new_tokens: int = Field(100, ge=1, le=2048, description="Max generated tokens")
    temperature: float = Field(0.8, ge=0.0, le=2.0, description="Sampling temperature")
    top_k: int = Field(40, ge=0, description="Top-k filtering threshold")
    top_p: float = Field(0.95, ge=0.0, le=1.0, description="Nucleus sampling threshold")
    repetition_penalty: float = Field(1.1, ge=1.0, le=3.0, description="Repetition penalty")


class GenerationResponse(BaseModel):
    prompt: str
    generated_text: str
    num_generated_tokens: int


class ModelInfoResponse(BaseModel):
    model_name: str
    total_parameters: int
    trainable_parameters: int
    vocab_size: int
    hidden_size: int
    num_layers: int
    device: str


def create_app(generator: Optional[TextGenerator] = None) -> "FastAPI":
    """Create and configure FastAPI application."""
    if not HAS_FASTAPI:
        raise ImportError("FastAPI is required. Install with `pip install fastapi uvicorn`")

    app = FastAPI(
        title="TounsiLM Inference API",
        description="Local inference server for Tunisian Arabic Language Model",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store generator reference in app state
    app.state.generator = generator

    @app.get("/health")
    def health():
        gen = app.state.generator
        return {
            "status": "healthy" if gen is not None else "degraded",
            "model_loaded": gen is not None,
            "device": str(gen.device) if gen is not None else "none",
        }

    @app.get("/model-info", response_model=ModelInfoResponse)
    def model_info():
        gen = app.state.generator
        if gen is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        counts = gen.model.count_parameters()
        cfg = gen.model.config
        return ModelInfoResponse(
            model_name="TounsiLM-300M",
            total_parameters=counts["total"],
            trainable_parameters=counts["trainable"],
            vocab_size=cfg.vocab_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            device=str(gen.device),
        )

    @app.post("/generate", response_model=GenerationResponse)
    def generate(req: GenerationRequest):
        gen = app.state.generator
        if gen is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        try:
            output = gen.generate(
                prompt=req.prompt,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
                repetition_penalty=req.repetition_penalty,
            )
            # Estimate generated tokens
            gen_tokens = len(gen.tokenizer.encode(output, add_bos=False, add_eos=False))
            return GenerationResponse(
                prompt=req.prompt,
                generated_text=output,
                num_generated_tokens=gen_tokens,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app
