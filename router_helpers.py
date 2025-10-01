# router_helpers.py
import os
import io
import re
import json
import time
import uuid
import mimetypes
import requests
import logging
from typing import Optional, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from google import genai
from google.genai import types, errors
from pydantic import BaseModel, Field

# ---------- Configuration ----------
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "")
_client = genai.Client(api_key=GEMINI_API_KEY)

def _to_int(val: str, default: int) -> int:
    try:
        return int(val)
    except Exception:
        return default

DECIDER_WORKERS = _to_int(os.getenv("DECIDER_WORKERS", "8"), 8)  # parallelism for decider
DECIDER_LOG_JSON = os.getenv("DECIDER_LOG_JSON", "0") in ("1", "true", "True", "yes", "YES")
_DECIDER_LOG_LEVEL = os.getenv("DECIDER_LOG_LEVEL", "INFO").upper()

logger = logging.getLogger("router_helpers.decider")
try:
    logger.setLevel(getattr(logging, _DECIDER_LOG_LEVEL, logging.INFO))
except Exception:
    logger.setLevel(logging.INFO)

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# ---------- Helpers ----------

def _safe_url(url: str, maxlen: int = 140) -> str:
    # Redact query params and clip
    try:
        base = url.split("?", 1)[0]
    except Exception:
        base = url
    return (base[:maxlen] + "...") if len(base) > maxlen else base

def _guess_mime(url: str, headers: requests.structures.CaseInsensitiveDict) -> str:
    """Prefer server content-type, then extension, default to JPEG."""
    ct = headers.get("Content-Type")
    if ct:
        return ct.split(";")[0].strip()
    typ, _ = mimetypes.guess_type(url)
    return typ or "image/jpeg"

def _part_from_url(url: str) -> types.Part:
    """Download image bytes and create a Part from bytes."""
    t0 = time.perf_counter()
    resp = requests.get(url, timeout=20, headers={"User-Agent": USER_AGENT})
    dt = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()
    mime = _guess_mime(url, resp.headers)
    size = len(resp.content)
    logger.debug(
        "downloaded image",
        extra={
            "event": "download_ok",
            "url": _safe_url(url),
            "status": resp.status_code,
            "mime": mime,
            "bytes": size,
            "ms": round(dt, 1),
        },
    )
    return types.Part.from_bytes(data=resp.content, mime_type=mime)

def _part_from_path(path: str) -> types.Part:
    """Create a Part from a local file path (bytes)."""
    with open(path, "rb") as f:
        data = f.read()
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    logger.debug(
        "loaded local image",
        extra={"event": "file_read_ok", "path": path, "bytes": len(data), "mime": mime},
    )
    return types.Part.from_bytes(data=data, mime_type=mime)

class RouterDecision(BaseModel):
    decision: str = Field(description="geographic or non-geographic")
    confidence: float = Field(description="0-1 score")
    reason: str = Field(description="brief rationale")

_DECISION_PUNCT_RE = re.compile(r"^[\s\.\!\?\"'\(\)\[\]\{\};:,\/\\\-_]+|[\s\.\!\?\"'\(\)\[\]\{\};:,\/\\\-_]+$")

def _normalize_decision(s: str) -> str:
    """
    Normalize model output to either 'geographic' or 'non-geographic'.
    Any non-exact match defaults to non-geographic (safer).
    """
    if not s:
        return "non-geographic"
    s = _DECISION_PUNCT_RE.sub("", str(s).strip().lower())
    s = s.replace("_", "-")
    if s in ("geographic", "geo", "geographical"):
        return "geographic"
    if s in ("non-geographic", "nongeographic", "non-geo", "nongeo", "not-geographic"):
        return "non-geographic"
    # Last chance: simple contains (very defensive)
    if "geograph" in s and "non" not in s:
        return "geographic"
    return "non-geographic"

# ---------- Model Calls ----------

def classify_pair_urls(front_url: str, back_url: Optional[str], _req_id: Optional[str] = None) -> dict:
    """
    Classify the postcard using ONLY the FRONT image.
    back_url is accepted for signature compatibility but ignored.
    """
    req_id = _req_id or str(uuid.uuid4())
    front_safe = _safe_url(front_url)

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
  or printed/handwritten text clearly naming a place.
- "Non-geographic": Holiday/seasonal greetings, novelty/comic art, generic people/animals/flowers,
  abstract/patterns, generic interiors, generic objects, and scenes with no clearly named place.

Rules:
- Use explicit place names or unmistakable landmark identity.
- If evidence is weak/ambiguous, prefer "non-geographic" and lower the confidence.
        """
    )

    logger.info(
        "decider_start",
        extra={"event": "decider_start", "req_id": req_id, "front_url": front_safe},
    )

    parts = [prompt, _part_from_url(front_url)]  # FRONT ONLY
    t0 = time.perf_counter()
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
        dt = (time.perf_counter() - t0) * 1000

        if DECIDER_LOG_JSON:
            logger.debug(
                "decider_raw_json",
                extra={"event": "decider_raw_json", "req_id": req_id, "json": resp.text},
            )

        # Parse and normalize
        try:
            parsed = json.loads(resp.text)
        except Exception as pe:
            logger.error(
                "decider_json_parse_error",
                extra={"event": "decider_json_parse_error", "req_id": req_id, "error": str(pe), "resp_text": resp.text[:500]},
            )
            parsed = {"decision": "non-geographic", "confidence": 0.0, "reason": "json_parse_error"}

        norm = _normalize_decision(parsed.get("decision", ""))
        parsed["decision"] = norm

        logger.info(
            "decider_done",
            extra={
                "event": "decider_done",
                "req_id": req_id,
                "ms": round(dt, 1),
                "decision": parsed.get("decision"),
                "confidence": parsed.get("confidence"),
                "reason": parsed.get("reason"),
            },
        )
        return parsed

    except errors.ClientError as e:
        dt = (time.perf_counter() - t0) * 1000
        logger.error(
            "decider_client_error",
            extra={"event": "decider_client_error", "req_id": req_id, "ms": round(dt, 1), "error": str(e)},
        )
        return {"decision": "non-geographic", "confidence": 0.0, "reason": f"gemini_error: {e}"}
    except Exception as e:
        dt = (time.perf_counter() - t0) * 1000
        logger.exception(
            "decider_exception",
            extra={"event": "decider_exception", "req_id": req_id, "ms": round(dt, 1), "error": str(e)},
        )
        return {"decision": "non-geographic", "confidence": 0.0, "reason": f"exception: {e}"}

# ---------- Routing ----------

def route_links(links: List[str]):
    """
    Returns (geo_links, nongeo_links, decisions)
      - geo_links / nongeo_links: flat lists of URLs in original pair order
      - decisions: list aligned to pair index:
          {"decision": "geographic"|"non-geographic", "confidence": float, "reason": str}
    """
    req_id = str(uuid.uuid4())
    n = len(links)
    logger.info(
        "routing_start",
        extra={"event": "routing_start", "req_id": req_id, "num_links": n, "decider_workers": DECIDER_WORKERS},
    )

    # Build (pair_index, front, back)
    pairs = []
    for i in range(0, n, 2):
        front = links[i]
        back = links[i + 1] if i + 1 < n else None
        pairs.append((i // 2, front, back))
        logger.debug(
            "pair_built",
            extra={
                "event": "pair_built",
                "req_id": req_id,
                "pair_index": i // 2,
                "front_url": _safe_url(front),
                "back_url": _safe_url(back) if back else None,
            },
        )

    decisions: List[Optional[dict]] = [None] * len(pairs)

    def _job(front_url, back_url):
        # We ALWAYS classify on the FRONT ONLY (no penalty if back is missing)
        return classify_pair_urls(front_url, back_url, _req_id=req_id)

    # Parallel classify
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=DECIDER_WORKERS) as ex:
        future_map = {ex.submit(_job, front, back): idx for (idx, front, back) in pairs}
        for fut in as_completed(future_map):
            idx = future_map[fut]
            try:
                decisions[idx] = fut.result()
            except Exception as e:
                logger.exception(
                    "decider_future_exception",
                    extra={"event": "decider_future_exception", "req_id": req_id, "pair_index": idx, "error": str(e)},
                )
                decisions[idx] = {
                    "decision": "non-geographic",
                    "confidence": 0.0,
                    "reason": f"decider error: {e}",
                }
    dt = (time.perf_counter() - t0) * 1000
    logger.info("classification_batch_done", extra={"event": "classification_batch_done", "req_id": req_id, "ms": round(dt, 1)})

    # Bucketize links using the decisions, preserving original pair order
    geo_links, nongeo_links = [], []
    geo_indices, nongeo_indices = [], []

    for (idx, front, back) in pairs:
        d = (decisions[idx] or {}).get("decision", "")
        d_norm = _normalize_decision(d)
        if d_norm == "geographic":
            bucket = geo_links
            geo_indices.append(idx)
        else:
            bucket = nongeo_links
            nongeo_indices.append(idx)

        if front:
            bucket.append(front)
        if back:
            bucket.append(back)

        logger.debug(
            "pair_bucketed",
            extra={
                "event": "pair_bucketed",
                "req_id": req_id,
                "pair_index": idx,
                "decision": d_norm,
                "front_url": _safe_url(front),
                "back_url": _safe_url(back) if back else None,
            },
        )

    logger.info(
        "routing_done",
        extra={
            "event": "routing_done",
            "req_id": req_id,
            "geo_pairs": len(geo_indices),
            "non_geo_pairs": len(nongeo_indices),
            "geo_indices": geo_indices,
            "non_geo_indices": nongeo_indices,
        },
    )

    return geo_links, nongeo_links, decisions  # keep raw decisions for logging/UI

# ---------- Local-file entrypoint ----------

def decide_label(front_path: str, back_path: Optional[str]) -> str:
    """
    Local-file entrypoint used elsewhere.
    **Uses only the FRONT image** for the decider.
    """
    req_id = str(uuid.uuid4())
    parts = [
        "Classify postcard as 'geographic' or 'non-geographic' (JSON with 'decision' only).",
        _part_from_path(front_path),  # FRONT ONLY
    ]
    try:
        t0 = time.perf_counter()
        resp = _client.models.generate_content(
            model="gemini-2.5-flash",
            contents=parts,
            config={
                "response_mime_type": "application/json",
                "response_schema": RouterDecision,
                "thinking_config": types.ThinkingConfig(thinking_budget=0),
            },
        )
        dt = (time.perf_counter() - t0) * 1000
        if DECIDER_LOG_JSON:
            logger.debug("decider_local_raw_json", extra={"event": "decider_local_raw_json", "req_id": req_id, "json": resp.text})

        d = _normalize_decision(json.loads(resp.text).get("decision", ""))
        logger.info("decider_local_done", extra={"event": "decider_local_done", "req_id": req_id, "decision": d, "ms": round(dt, 1)})
        return d
    except Exception as e:
        logger.exception("decider_local_exception", extra={"event": "decider_local_exception", "req_id": req_id, "error": str(e)})
        return "non-geographic"
