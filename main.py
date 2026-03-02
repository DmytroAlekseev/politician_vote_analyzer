#!/usr/bin/env python3
"""
MEP Vote Analyzer — Flask Backend v2
Data source: HowTheyVote.eu CSV exports (GitHub releases)

Run:
    pip install flask requests pandas
    python app.py

Then open: http://localhost:5000

CSVs are cached in ./cache/ after first download (~30 MB total).
Use ?refresh=1 in any /api/search call to re-download.
"""

import io
import os
from pathlib import Path

import pandas as pd
import requests
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder="static")

# ── Download URLs ─────────────────────────────────────────────────────────────
BASE = "https://github.com/HowTheyVote/data/releases/latest/download"
URLS = {
    "members":      f"{BASE}/members.csv.gz",
    "votes":        f"{BASE}/votes.csv.gz",
    "member_votes": f"{BASE}/member_votes.csv.gz",
    "groups":       f"{BASE}/groups.csv.gz",
    "group_memberships": f"{BASE}/group_memberships.csv.gz",
}

CACHE_DIR = Path("cache")
_data: dict = {}


# ── Data loading ──────────────────────────────────────────────────────────────

USE_DISK = os.environ.get("RENDER") is None  # local = use disk cache; Render = memory only

def _download(key: str, force: bool = False) -> pd.DataFrame:
    if USE_DISK:
        path = CACHE_DIR / f"{key}.csv.gz"
        if force and path.exists():
            path.unlink()
        if not path.exists():
            print(f"  Downloading {key}.csv.gz …", flush=True)
            r = requests.get(URLS[key], timeout=90, headers={"User-Agent": "MEP-Analyzer/2.0"})
            r.raise_for_status()
            path.write_bytes(r.content)
        else:
            print(f"  Cached      {key}.csv.gz", flush=True)
        df = pd.read_csv(path, compression="gzip", low_memory=False)
    else:
        # On Render free tier: stream directly into memory, no disk write
        print(f"  Downloading {key}.csv.gz into memory …", flush=True)
        r = requests.get(URLS[key], timeout=90, headers={"User-Agent": "MEP-Analyzer/2.0"})
        r.raise_for_status()
        df = pd.read_csv(io.BytesIO(r.content), compression="gzip", low_memory=False)
    df.columns = df.columns.str.strip()
    return df


def load_data(force: bool = False) -> None:
    global _data
    CACHE_DIR.mkdir(exist_ok=True)
    print("\nLoading dataset …")

    members      = _download("members", force)
    votes        = _download("votes", force)
    member_votes = _download("member_votes", force)
    groups       = _download("groups", force)

    # Build full_name
    first = members.get("first_name", pd.Series("", index=members.index)).fillna("")
    last  = members.get("last_name",  pd.Series("", index=members.index)).fillna("")
    members["full_name"] = (first + " " + last).str.strip()

    # Build group lookup: code → short_label
    group_map = dict(zip(groups["code"], groups.get("short_label", groups["code"])))

    _data = {
        "members":      members,
        "votes":        votes,
        "member_votes": member_votes,
        "group_map":    group_map,
    }
    print(f"  ✓ {len(members):,} MEPs | {len(votes):,} votes | {len(member_votes):,} individual votes\n")


# ── Helpers ───────────────────────────────────────────────────────────────────

def safe(val):
    """Convert NaN / NaT to None for JSON serialisation."""
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    return val


def to_int(val):
    try:
        v = int(val)
        return v if v >= 0 else None
    except (TypeError, ValueError):
        return None


# ── Startup data load (runs under both gunicorn and direct python) ───────────
import atexit as _atexit

@app.before_request
def _ensure_loaded():
    if not _data:
        try:
            load_data()
        except Exception as e:
            pass  # will surface as error in the search endpoint

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/search")
def search():
    mep_query   = request.args.get("mep",   "").strip()
    topic_query = request.args.get("topic", "").strip()
    refresh     = request.args.get("refresh", "0") == "1"

    if not mep_query or not topic_query:
        return jsonify({"error": "Both 'mep' and 'topic' parameters are required."}), 400

    if not _data or refresh:
        try:
            load_data(force=refresh)
        except Exception as e:
            return jsonify({"error": f"Failed to load dataset: {e}"}), 502

    members      = _data["members"]
    votes        = _data["votes"]
    member_votes = _data["member_votes"]
    group_map    = _data["group_map"]

    # ── 1. Find MEP ──────────────────────────────────────────────────────────
    q    = mep_query.lower()
    mask = members["full_name"].str.lower().str.contains(q, na=False)
    hits = members[mask]

    if hits.empty:
        return jsonify({"error": f"No MEP found matching '{mep_query}'. Try a partial last name."}), 404

    mep      = hits.iloc[0]
    mep_id   = mep["id"]
    mep_name = mep["full_name"]
    all_matches = hits["full_name"].tolist()

    # ── 2. Find votes matching topic ─────────────────────────────────────────
    tq = topic_query.lower()
    title_col = "display_title" if "display_title" in votes.columns else "title"

    masks = [votes[title_col].str.lower().str.contains(tq, na=False)]
    for col in ("description", "procedure_title", "amendment_subject"):
        if col in votes.columns:
            masks.append(votes[col].str.lower().str.contains(tq, na=False))

    combined_mask = masks[0]
    for m in masks[1:]:
        combined_mask = combined_mask | m

    topic_votes = votes[combined_mask].copy()

    if topic_votes.empty:
        return jsonify({"error": f"No votes found matching topic '{topic_query}'."}), 404

    # ── 3. Cross-reference with this MEP's individual votes ──────────────────
    vote_ids  = set(topic_votes["id"].tolist())
    mep_mv    = member_votes[
        (member_votes["member_id"] == mep_id) &
        (member_votes["vote_id"].isin(vote_ids))
    ].copy()

    if mep_mv.empty:
        return jsonify({
            "error": (
                f"{mep_name} has no recorded roll-call votes on '{topic_query}'. "
                "Only roll-call votes are tracked; show-of-hands votes are not."
            )
        }), 404

    # Merge vote metadata into mep_mv
    keep_cols = ["id", title_col, "timestamp", "description",
                 "amendment_subject", "amendment_number",
                 "is_main", "reference",
                 "procedure_reference", "procedure_title", "procedure_type",
                 "procedure_stage", "result",
                 "count_for", "count_against", "count_abstention", "count_did_not_vote"]
    keep_cols = [c for c in keep_cols if c in topic_votes.columns]

    merged = mep_mv.merge(topic_votes[keep_cols], left_on="vote_id", right_on="id", how="left")

    # ── 4. Tally & build response ─────────────────────────────────────────────
    # Percentages are calculated from MAIN votes only (final resolutions).
    # All votes (incl. amendments) are still returned for the detail list.
    main_counts = {"FOR": 0, "AGAINST": 0, "ABSTENTION": 0, "DID_NOT_VOTE": 0}
    all_counts  = {"FOR": 0, "AGAINST": 0, "ABSTENTION": 0, "DID_NOT_VOTE": 0}
    result_votes = []

    for _, row in merged.iterrows():
        pos = str(row.get("position", "")).strip().upper()
        if pos not in all_counts:
            pos = "ABSTENTION"
        all_counts[pos] += 1
        is_main = bool(row.get("is_main")) if safe(row.get("is_main")) is not None else False
        if is_main:
            main_counts[pos] += 1

        # Group the MEP was in on this vote
        gc = safe(row.get("group_code"))
        group_label = group_map.get(gc, gc) if gc else None

        # Parliament totals for this vote
        cf  = to_int(row.get("count_for"))
        ca  = to_int(row.get("count_against"))
        cab = to_int(row.get("count_abstention"))
        cdnv = to_int(row.get("count_did_not_vote"))

        result_votes.append({
            "vote_id":           int(row["vote_id"]),
            "date":              str(row[title_col] and row.get("timestamp", ""))[:10] if safe(row.get("timestamp")) else "",
            "timestamp":         safe(row.get("timestamp")),
            "title":             str(safe(row.get(title_col)) or "Untitled")[:200],
            "description":       safe(row.get("description")),
            "amendment_subject": safe(row.get("amendment_subject")),
            "amendment_number":  safe(row.get("amendment_number")),
            "is_main":           bool(row.get("is_main")) if safe(row.get("is_main")) is not None else None,
            "reference":         safe(row.get("reference")),
            "procedure_reference": safe(row.get("procedure_reference")),
            "procedure_title":   safe(row.get("procedure_title")),
            "procedure_type":    safe(row.get("procedure_type")),
            "procedure_stage":   safe(row.get("procedure_stage")),
            "result":            safe(row.get("result")),
            "position":          pos,
            "group":             group_label,
            "parliament_for":    cf,
            "parliament_against": ca,
            "parliament_abstention": cab,
            "parliament_did_not_vote": cdnv,
            "parliament_total":  sum(x for x in [cf, ca, cab, cdnv] if x is not None) or None,
        })

    # Sort: main votes first, then newest first within each group
    result_votes.sort(
        key=lambda v: (0 if v.get("is_main") else 1, str(v.get("timestamp") or "")),
        reverse=True
    )
    result_votes.sort(key=lambda v: 0 if v.get("is_main") else 1)

    total       = len(result_votes)
    main_total  = sum(main_counts.values())

    # Fall back to all votes if no main votes exist (older data may lack is_main)
    counts_for_pct = main_counts if main_total > 0 else all_counts
    total_for_pct  = main_total  if main_total > 0 else total

    return jsonify({
        "mep_name":        mep_name,
        "mep_id":          int(mep_id),
        "topic":           topic_query,
        "total":           total,
        "main_total":      main_total,
        "counts":          counts_for_pct,
        "all_counts":      all_counts,
        "percentages":     {k: round(v / total_for_pct * 100, 1) for k, v in counts_for_pct.items()},
        "pct_based_on":    "main" if main_total > 0 else "all",
        "votes":           result_votes,
        "all_matches":     all_matches,
        "multiple":        len(all_matches) > 1,
    })


@app.route("/api/status")
def status():
    loaded = bool(_data)
    return jsonify({
        "loaded":  loaded,
        "members": len(_data["members"]) if loaded else 0,
        "votes":   len(_data["votes"])   if loaded else 0,
    })


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    Path("static").mkdir(exist_ok=True)
    load_data()
    port = int(os.environ.get("PORT", 5000))
    local = os.environ.get("RENDER") is None
    if local:
        import webbrowser, threading
        print("  MEP Vote Analyzer")
        print(f"  Running at → http://localhost:{port}")
        print("  Press Ctrl+C to stop\n")
        threading.Timer(1.2, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    app.run(debug=False, host="0.0.0.0", port=port)