# router_helpers.py
import json
from typing import List, Tuple, Dict
from google import genai
from mimetypes import guess_type
from google.genai import types

# Reuse the same API key your app already uses
import os
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "")
_client = genai.Client(api_key=GEMINI_API_KEY)

# Minimal JSON schema for the decider
_DECIDER_SCHEMA = types.Schema(
    type="object",
    properties={
        "decision": types.Schema(type="string", enum=["geographic", "non-geographic"]),
        "confidence": types.Schema(type="number"),
        "reason": types.Schema(type="string"),
    },
    required=["decision", "confidence"]
)

_DEF_PROMPT = """Decide if this postcard pair is GEOGRAPHIC or NON-GEOGRAPHIC.

GEOGRAPHIC = The main subject is a place you can point to on a map
(city, town, park, street, bridge, building, landmark, etc.), or the
back text/address/postmark clearly indicates a place to list. Think
"view cards": street scenes, city halls, depots, beaches, bridges, etc.

NON-GEOGRAPHIC = Holiday/holiday motifs, greetings, animals, children,
romance, humor, fantasy/illustration, topical art, etc., where no
specific place is the main subject to list.

Return JSON only:
{"decision":"geographic|non-geographic","confidence":0..1,"reason":"<short>"}"""

def _part_from_path(p: str) -> types.Part:
    with open(p, "rb") as f:
        return types.Part.from_bytes(f.read(), mime_type="image/jpeg")

def _part_from_url(u: str):
    mt, _ = guess_type(u)
    if not mt:
        mt = "image/jpeg"  # sensible default
    return types.Part.from_uri(file_uri=u, mime_type=mt)

def classify_pair_paths(front_path: str, back_path: str) -> Dict:
    """Classify a local file pair. Returns a dict with keys decision/confidence/reason."""
    front = _part_from_path(front_path)
    back  = _part_from_path(back_path)
    resp = _client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[_DEF_PROMPT, front, back],
        config={
            "response_mime_type": "application/json",
            "response_schema": _DECIDER_SCHEMA,
            # Disable “thinking”/chain-of-thought; we only want the JSON decision:
            "thinking_config": types.ThinkingConfig(thinking_budget=0)
        },
    )
    return json.loads(resp.text)

def classify_pair_urls(front_url: str, back_url: str) -> Dict:
    """Classify a URL pair. Returns a dict with keys decision/confidence/reason."""
    front = _part_from_url(front_url)
    back  = _part_from_url(back_url)
    resp = _client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[_DEF_PROMPT, front, back],
        config={
            "response_mime_type": "application/json",
            "response_schema": _DECIDER_SCHEMA,
            "thinking_config": types.ThinkingConfig(thinking_budget=0)
        },
    )
    return json.loads(resp.text)

def decide_label(front_path: str, back_path: str) -> str:
    """Compatibility wrapper used by existing worker code (local paths)."""
    try:
        out = classify_pair_paths(front_path, back_path)
        return "geographic" if out.get("decision") == "geographic" else "non-geographic"
    except Exception:
        return "non-geographic"

def route_links(links: List[str]) -> Tuple[List[str], List[str], List[Dict]]:
    """
    Given a flat list of alternating front/back URLs, split into
    geographic and non-geographic link lists (keeps original order).
    Also returns a per-pair decision record for audit.
    """
    geo, nongeo, audit = [], [], []
    for i in range(0, len(links), 2):
        front = links[i]
        back = links[i+1] if i+1 < len(links) else None
        result = classify_pair_urls(front, back) if back else {"decision": "non-geographic", "confidence": 0, "reason": "No back image"}
        (geo if result["decision"] == "geographic" else nongeo).extend([front] + ([back] if back else []))
        audit.append({"pair_index": i//2, "front": front, "back": back, **result})
    return geo, nongeo, audit
