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
from typing import Optional, List, Dict, Any
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
# Ensure our messages propagate to root handlers
logger.propagate = True
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
    if not url:
        return ""
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
    logger.debug(f"[decider] download_ok url={_safe_url(url)} status={resp.status_code} mime={mime} bytes={size} ms={round(dt,1)}")
    return types.Part.from_bytes(data=resp.content, mime_type=mime)

def _part_from_path(path: str) -> types.Part:
    """Create a Part from a local file path (bytes)."""
    with open(path, "rb") as f:
        data = f.read()
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    logger.debug(f"[decider] file_read_ok path={path} bytes={len(data)} mime={mime}")
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
    if "geograph" in s and "non" not in s:
        return "geographic"
    return "non-geographic"

# ---------- Model Calls (FRONT ONLY) ----------

def classify_pair_urls(front_url: str, back_url: Optional[str], _req_id: Optional[str] = None, pair_index: Optional[int] = None) -> Dict[str, Any]:
    """
    Classify the postcard using ONLY the FRONT image.
    back_url is accepted for signature compatibility but ignored.
    """
    req_id = _req_id or uuid.uuid4().hex[:8]
    pidx = -1 if pair_index is None else pair_index
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
    ).strip()

    logger.info(f"[decider] start req_id={req_id} pair_index={pidx} front={front_safe} (BACK IGNORED)")

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
            logger.debug(f"[decider] raw_json req_id={req_id} pair_index={pidx} json={resp.text}")

        # Parse and normalize
        try:
            parsed = json.loads(resp.text)
        except Exception as pe:
            logger.error(f"[decider] json_parse_error req_id={req_id} pair_index={pidx} err={pe} resp_snip={resp.text[:500]}")
            parsed = {"decision": "non-geographic", "confidence": 0.0, "reason": "json_parse_error"}

        norm = _normalize_decision(parsed.get("decision", ""))
        parsed["decision"] = norm

        logger.info(
            f"[decider] done req_id={req_id} pair_index={pidx} ms={round(dt,1)} "
            f"decision={parsed.get('decision')} conf={parsed.get('confidence')} reason=\"{parsed.get('reason')}\""
        )
        return parsed

    except errors.ClientError as e:
        dt = (time.perf_counter() - t0) * 1000
        logger.error(f"[decider] client_error req_id={req_id} pair_index={pidx} ms={round(dt,1)} err={e}")
        return {"decision": "non-geographic", "confidence": 0.0, "reason": f"gemini_error: {e}"}
    except Exception as e:
        dt = (time.perf_counter() - t0) * 1000
        logger.exception(f"[decider] exception req_id={req_id} pair_index={pidx} ms={round(dt,1)} err={e}")
        return {"decision": "non-geographic", "confidence": 0.0, "reason": f"exception: {e}"}

# ---------- Routing ----------

def route_links(links: List[str]):
    """
    Returns (geo_links, nongeo_links, decisions)
      - geo_links / nongeo_links: flat lists of URLs in original pair order
      - decisions: list aligned to pair index:
          {"decision": "geographic"|"non-geographic", "confidence": float, "reason": str}
    """
    req_id = uuid.uuid4().hex[:8]
    n = len(links)
    pairs_ct = n // 2
    logger.info(f"[decider] routing_start req_id={req_id} links={n} pairs={pairs_ct} workers={DECIDER_WORKERS}")

    # Build (pair_index, front, back)
    pairs: List[tuple] = []
    for i in range(0, n, 2):
        front = links[i]
        back = links[i + 1] if i + 1 < n else None
        pairs.append((i // 2, front, back))
        logger.debug(f"[decider] pair_built req_id={req_id} pair_index={i//2} front={_safe_url(front)} back={_safe_url(back) if back else None}")

    decisions: List[Optional[dict]] = [None] * len(pairs)

    def _job(front_url, back_url, idx):
        # FRONT ONLY classification
        return classify_pair_urls(front_url, back_url, _req_id=req_id, pair_index=idx)

    # Parallel classify
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=DECIDER_WORKERS) as ex:
        future_map = {ex.submit(_job, front, back, idx): idx for (idx, front, back) in pairs}
        for fut in as_completed(future_map):
            idx = future_map[fut]
            try:
                decisions[idx] = fut.result()
            except Exception as e:
                logger.exception(f"[decider] future_exception req_id={req_id} pair_index={idx} err={e}")
                decisions[idx] = {
                    "decision": "non-geographic",
                    "confidence": 0.0,
                    "reason": f"decider error: {e}",
                }
    dt = (time.perf_counter() - t0) * 1000
    logger.info(f"[decider] classification_batch_done req_id={req_id} ms={round(dt,1)}")

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
            f"[decider] pair_bucketed req_id={req_id} pair_index={idx} decision={d_norm} "
            f"front={_safe_url(front)} back={_safe_url(back) if back else None}"
        )

    logger.info(
        f"[decider] routing_done req_id={req_id} geo_pairs={len(geo_indices)} non_geo_pairs={len(nongeo_indices)} "
        f"geo_indices={geo_indices} non_geo_indices={nongeo_indices}"
    )

    return geo_links, nongeo_links, decisions  # keep raw decisions for logging/UI

# ---------- Local-file entrypoint ----------

def decide_label(front_path: str, back_path: Optional[str]) -> str:
    """
    Local-file entrypoint used elsewhere.
    **Uses only the FRONT image** for the decider.
    """
    req_id = uuid.uuid4().hex[:8]
    parts = [
        "Classify postcard as 'geographic' or 'non-geographic' (JSON with 'decision','confidence','reason').",
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
            logger.debug(f"[decider] local_raw_json req_id={req_id} json={resp.text}")

        data = json.loads(resp.text or "{}")
        decision = _normalize_decision(data.get("decision", ""))
        conf = float(data.get("confidence", 0.0)) if isinstance(data.get("confidence", 0.0), (int, float, str)) else 0.0
        reason = str(data.get("reason", "")).strip()

        logger.info(f"[decider] local_done req_id={req_id} ms={round(dt,1)} decision={decision} conf={conf} reason=\"{reason}\"")
        return decision
    except Exception as e:
        logger.exception(f"[decider] local_exception req_id={req_id} err={e}")
        return "non-geographic"
