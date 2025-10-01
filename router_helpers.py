# router_helpers.py (drop-in replacement for URL parts & classifier bits)
import os, io, json, mimetypes, requests, logging
from typing import Optional, Tuple, List
from google import genai
from google.genai import types, errors
import os
from concurrent.futures import ThreadPoolExecutor, as_completed


GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "")
_client = genai.Client(api_key=GEMINI_API_KEY)

DECIDER_WORKERS = int(os.getenv("DECIDER_WORKERS", "8"))  # default=8

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

def _guess_mime(url: str, headers: requests.structures.CaseInsensitiveDict) -> str:
    # Prefer server-provided content type; fall back to extension; default to jpeg
    ct = headers.get("Content-Type")
    if ct:
        return ct.split(";")[0].strip()
    typ, _ = mimetypes.guess_type(url)
    return typ or "image/jpeg"

def _part_from_url(url: str) -> types.Part:
    # Download bytes, then construct a Part from bytes (NOT from_uri)
    resp = requests.get(url, timeout=20, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    mime = _guess_mime(url, resp.headers)
    return types.Part.from_bytes(data=resp.content, mime_type=mime)

# Optional: if you want to support local file paths as well
def _part_from_path(path: str) -> types.Part:
    with open(path, "rb") as f:
        data = f.read()
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    return types.Part.from_bytes(data=data, mime_type=mime)

# Pydantic-like schema for the response (if you want the SDK to validate JSON)
# You can also keep using a literal schema dict or parse resp.text directly.
from pydantic import BaseModel, Field
class RouterDecision(BaseModel):
    decision: str = Field(description="geographic or non-geographic")
    confidence: float = Field(description="0-1 score")
    reason: str = Field(description="brief rationale")

def classify_pair_urls(front_url: str, back_url: Optional[str]) -> dict:
    prompt = ("""
        "Decide if this postcard (front and optionally back) is GEOGRAPHIC "
        "(clearly tied to a place/city/state/landmark) or NON-GEOGRAPHIC. "
        "Respond as JSON with fields: decision ('geographic'|'non-geographic'), "
        "confidence (0..1), reason (short)."
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

        """
    )
    parts = [prompt, _part_from_url(front_url)]

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
        print(json.loads(resp.text))
        return json.loads(resp.text)

    except errors.ClientError as e:
        logging.error(f"Gemini classify error: {e}")
        # Fallback: mark as non-geographic so it doesn't block your pipeline
        return {"decision": "non-geographic", "confidence": 0.0, "reason": f"gemini_error: {e}"}

def route_links(links):
    """
    Returns (geo_links, nongeo_links, decisions)
    - geo_links / nongeo_links are flat lists of URLs (front, back, front, back, ...)
    - decisions is a list aligned to pair index: {"decision": "geographic"|"non-geographic", "confidence": float, "reason": str}
    """
    # Build (pair_index, front, back)
    pairs = []
    for i in range(0, len(links), 2):
        front = links[i]
        back = links[i + 1] if i + 1 < len(links) else None
        pairs.append((i // 2, front, back))

    decisions = [None] * len(pairs)

    # --- Parallel classify: up to DECIDER_WORKERS at once ---
    def _job(front_url, back_url):
        # If you accept singletons, you can decide a default here
        if not back_url:
            return {"decision": "non-geographic", "confidence": 0.0, "reason": "No back image"}
        return classify_pair_urls(front_url, back_url)

    with ThreadPoolExecutor(max_workers=DECIDER_WORKERS) as ex:
        future_map = {ex.submit(_job, front, back): idx for (idx, front, back) in pairs}
        for fut in as_completed(future_map):
            idx = future_map[fut]
            try:
                decisions[idx] = fut.result()
            except Exception as e:
                # Fail safe to non-geo on errors so the pipeline keeps moving
                decisions[idx] = {
                    "decision": "non-geographic",
                    "confidence": 0.0,
                    "reason": f"decider error: {e}"
                }

    # Bucketize links using the decisions, preserving original pair order
    geo_links, nongeo_links = [], []
    for (idx, front, back) in pairs:
        bucket = geo_links if decisions[idx].get("decision") == "geographic" else nongeo_links
        if front:
            bucket.append(front)
        if back:
            bucket.append(back)

    return geo_links, nongeo_links, decisions


# If you also expose a local-file entrypoint used elsewhere:
def decide_label(front_path: str, back_path: Optional[str]) -> str:
    parts = [
        "Classify postcard as 'geographic' or 'non-geographic' (JSON with 'decision' only).",
        _part_from_path(front_path),
    ]
    if back_path:
        parts.append(_part_from_path(back_path))
    resp = _client.models.generate_content(
        model="gemini-2.5-flash",
        contents=parts,
        config={
            "response_mime_type": "application/json",
            "response_schema": RouterDecision,
            "thinking_config": types.ThinkingConfig(thinking_budget=0),
        },
    )
    return json.loads(resp.text).get("decision", "non-geographic")
