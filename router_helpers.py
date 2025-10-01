# router_helpers.py
import os
import io
import json
import mimetypes
import requests
import logging
from typing import Optional, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from google import genai
from google.genai import types, errors

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "")
_client = genai.Client(api_key=GEMINI_API_KEY)

DECIDER_WORKERS = int(os.getenv("DECIDER_WORKERS", "8"))  # default parallelism = 8

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def _guess_mime(url: str, headers: requests.structures.CaseInsensitiveDict) -> str:
    """Prefer server content-type, then file extension, default to JPEG."""
    ct = headers.get("Content-Type")
    if ct:
        return ct.split(";")[0].strip()
    typ, _ = mimetypes.guess_type(url)
    return typ or "image/jpeg"


def _part_from_url(url: str) -> types.Part:
    """Download image bytes and create a Part from bytes."""
    resp = requests.get(url, timeout=20, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    mime = _guess_mime(url, resp.headers)
    return types.Part.from_bytes(data=resp.content, mime_type=mime)


def _part_from_path(path: str) -> types.Part:
    """Create a Part from a local file path (bytes)."""
    with open(path, "rb") as f:
        data = f.read()
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    return types.Part.from_bytes(data=data, mime_type=mime)


# Pydantic schema to have the SDK structure the JSON response
from pydantic import BaseModel, Field


class RouterDecision(BaseModel):
    decision: str = Field(description="geographic or non-geographic")
    confidence: float = Field(description="0-1 score")
    reason: str = Field(description="brief rationale")


def classify_pair_urls(front_url: str, back_url: Optional[str]) -> dict:
    """
    Classify the postcard using ONLY the FRONT image (by design).
    The back_url is accepted for signature compatibility but ignored.
    """
    prompt = (
        """
Decide if this postcard is GEOGRAPHIC (clearly tied to a place/city/state/country/landmark)
or NON-GEOGRAPHIC. Respond as JSON with fields:
- decision: "geographic" | "non-geographic"
- confidence: number in [0,1]
- reason: short rationale

Definitions:
- "Geographic": The primary subject is a real-world place or feature: city/town, street/bridge,
  named building, skyline, neighborhood, park, national/state/local landmark, mountain, river,
  lake, or other addressable location. Clues include printed captions, signage, publisher lines,
  or handwritten/printed text clearly naming a place.
- "Non-geographic": Holiday/seasonal greetings, novelty/comic art, generic people/animals/flowers,
  abstract/patterns, generic interiors, generic objects, and scenes with no clearly named place.

Rules:
- Use explicit place names or unmistakable landmark identity.
- If evidence is weak/ambiguous, prefer "non-geographic" and lower the confidence.
        """
    )

    parts = [prompt, _part_from_url(front_url)]  # FRONT ONLY

    try:
        resp = _client.models.generate_content(
            model="gemini-2.5-flash",
            contents=parts,
            config={
                "response_mime_type": "application/json",
                "response_schema": RouterDecision,
                "thinking_config": types.ThinkingConfig(thinking_budget=0),
            },
        )
        return json.loads(resp.text)
    except errors.ClientError as e:
        logging.error(f"Gemini classify error: {e}")
        # Fail-safe so pipeline keeps moving
        return {"decision": "non-geographic", "confidence": 0.0, "reason": f"gemini_error: {e}"}
    except Exception as e:
        logging.exception(f"Unexpected classify error for {front_url}: {e}")
        return {"decision": "non-geographic", "confidence": 0.0, "reason": f"exception: {e}"}


def route_links(links: List[str]):
    """
    Returns (geo_links, nongeo_links, decisions)
      - geo_links / nongeo_links: flat lists of URLs in original pair order
      - decisions: list aligned to pair index:
          {"decision": "geographic"|"non-geographic", "confidence": float, "reason": str}
    """
    # Build (pair_index, front, back)
    pairs = []
    for i in range(0, len(links), 2):
        front = links[i]
        back = links[i + 1] if i + 1 < len(links) else None
        pairs.append((i // 2, front, back))

    decisions: List[Optional[dict]] = [None] * len(pairs)

    # --- Parallel classify: up to DECIDER_WORKERS at once ---
    def _job(front_url, back_url):
        # We ALWAYS classify on the FRONT ONLY (no penalty if back is missing)
        return classify_pair_urls(front_url, back_url)

    with ThreadPoolExecutor(max_workers=DECIDER_WORKERS) as ex:
        future_map = {ex.submit(_job, front, back): idx for (idx, front, back) in pairs}
        for fut in as_completed(future_map):
            idx = future_map[fut]
            try:
                decisions[idx] = fut.result()
            except Exception as e:
                decisions[idx] = {
                    "decision": "non-geographic",
                    "confidence": 0.0,
                    "reason": f"decider error: {e}",
                }

    # Bucketize links using the decisions, preserving original pair order
    geo_links, nongeo_links = [], []
    for (idx, front, back) in pairs:
        d = (decisions[idx] or {}).get("decision", "")
        d_norm = str(d).strip().lower().replace("_", "-")  # tolerant routing
        bucket = geo_links if d_norm == "geographic" else nongeo_links
        if front:
            bucket.append(front)
        if back:
            bucket.append(back)

    return geo_links, nongeo_links, decisions  # keep raw decisions for logging/UI


def decide_label(front_path: str, back_path: Optional[str]) -> str:
    """
    Local-file entrypoint used elsewhere.
    **Uses only the FRONT image** for the decider.
    """
    parts = [
        "Classify postcard as 'geographic' or 'non-geographic' (JSON with 'decision' only).",
        _part_from_path(front_path),  # FRONT ONLY
    ]

    try:
        resp = _client.models.generate_content(
            model="gemini-2.5-flash",
            contents=parts,
            config={
                "response_mime_type": "application/json",
                "response_schema": RouterDecision,
                "thinking_config": types.ThinkingConfig(thinking_budget=0),
            },
        )
        d = json.loads(resp.text).get("decision", "non-geographic")
        return str(d).strip().lower().replace("_", "-")
    except Exception as e:
        logging.exception(f"decide_label error: {e}")
        return "non-geographic"
