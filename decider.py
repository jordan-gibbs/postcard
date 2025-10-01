#!/usr/bin/env python3
"""
Geographic vs Non-Geographic postcard decider (Gemini)

- Strict JSON output
- Uses response_schema to enforce structure
- Thinking disabled (thinking_budget=0) for minimal latency and cost
"""

from __future__ import annotations

import argparse
import io
import json
import mimetypes
import os
from typing import Literal, Optional, List

import requests
from PIL import Image
from pydantic import BaseModel, Field
from google import genai
from google.genai import types


# ---------- Configuration ----------

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "")
if not GEMINI_API_KEY:
    raise RuntimeError("Set GOOGLE_API_KEY in your environment.")

# IMPORTANT: Use a Flash model so thinking can be set to 0 (Pro does not support turning it off)
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

client = genai.Client(api_key=GEMINI_API_KEY)


# ---------- Schema enforced JSON ----------

class GeoDecision(BaseModel):
    """JSON output schema."""
    decision: Literal["geographic", "non-geographic"] = Field(
        description="Binary decision for the postcard pair."
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Model confidence from 0.0 to 1.0."
    )
    signals: Optional[List[str]] = Field(
        default=None,
        description="Short bullet signals that led to the decision."
    )


# ---------- Prompt ----------

PROMPT = """You are a strict binary classifier for postcard PAIRS (front and back).

Definition:
- "Geographic": The primary subject is a real-world place or feature: city/town, street/bridge,
  named building, skyline, neighborhood, park, national/state/local landmark, mountain, river,
  lake, or other addressable location. Clues include printed captions, signage, publisher lines
  or handwritten/printed text clearly naming a place.
- "Non-geographic": Holiday/seasonal greetings, novelty/comic art, generic people/animals/flowers,
  abstract/patterns, generic interiors, generic objects, and scenes with no clearly named place.

Rules:
- If EITHER the front OR back provides strong evidence of a specific named place, classify as "geographic".
- If evidence is weak/ambiguous, prefer "non-geographic" and lower the confidence.
- Do NOT infer places from vague hints; require explicit place names or unmistakable landmark identity.

Return ONLY JSON that matches this schema:
{
  "decision": "geographic" | "non-geographic",
  "confidence": <float 0..1>,
  "signals": [ "short reason 1", "short reason 2" ]
}
"""


# ---------- Helpers ----------

def _guess_mime(path: str, data: bytes) -> str:
    mime, _ = mimetypes.guess_type(path)
    if mime:
        return mime
    # Fallback to PIL sniffing
    try:
        img = Image.open(io.BytesIO(data))
        fmt = (img.format or "").lower()
        if fmt == "jpg":
            fmt = "jpeg"
        return f"image/{fmt}" if fmt else "application/octet-stream"
    except Exception:
        return "application/octet-stream"


def _to_part(source: str) -> types.Part:
    """
    Accepts a local path or HTTP(S) URL and returns a Part for multimodal input.
    """
    if source.startswith(("http://", "https://")):
        resp = requests.get(source, timeout=45)
        resp.raise_for_status()
        data = resp.content
        mime = resp.headers.get("Content-Type") or _guess_mime(source, data)
    else:
        with open(source, "rb") as f:
            data = f.read()
        mime = _guess_mime(source, data)
    return types.Part.from_bytes(data=data, mime_type=mime)


# ---------- Classifier ----------

def classify_pair(front_image: str, back_image: str) -> dict:
    """
    :param front_image: path or URL to the FRONT image
    :param back_image:  path or URL to the BACK image
    :return: dict with keys: decision, confidence, signals
    """
    parts = [
        types.Part.from_text(PROMPT),
        _to_part(front_image),
        _to_part(back_image),
    ]

    cfg = types.GenerateContentConfig(
        # Enforce JSON output and schema
        response_mime_type="application/json",
        response_schema=GeoDecision,
        # Turn off thinking for 2.5 Flash-family models
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        # Make behavior deterministic/repeatable
        temperature=0,
        max_output_tokens=128,
    )

    resp = client.models.generate_content(
        model=MODEL,
        contents=parts,
        config=cfg,
    )

    # The SDK returns .text already as a JSON string per response_mime_type
    try:
        return json.loads(resp.text)
    except Exception:
        # Safe fallback
        return {
            "decision": "non-geographic",
            "confidence": 0.0,
            "signals": ["parse_error"],
            "raw": resp.text,
        }


# ---------- CLI ----------

def _main():
    parser = argparse.ArgumentParser(
        description="Classify a postcard pair as geographic vs non-geographic"
    )
    parser.add_argument("--front", required=True, help="Front image path or URL")
    parser.add_argument("--back", required=True, help="Back image path or URL")
    args = parser.parse_args()

    result = classify_pair(args.front, args.back)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    _main()
