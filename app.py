#!/usr/bin/env python3
"""
MEP Vote Analyzer - Flask Backend v4
Memory-optimised for Render free tier (512 MB RAM).
Only main votes (final resolutions) are loaded - amendments are ignored.
"""

import io, os
from pathlib import Path

import pandas as pd
import requests
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder="static")

BASE = "https://github.com/HowTheyVote/data/releases/latest/download"
URLS = {
    "members":      f"{BASE}/members.csv.gz",
    "votes":        f"{BASE}/votes.csv.gz",
    "member_votes": f"{BASE}/member_votes.csv.gz",
    "groups":       f"{BASE}/groups.csv.gz",
}

CACHE_DIR = Path("cache")
USE_DISK  = os.environ.get("RENDER") is None
_data: dict = {}


def _get_raw(key: str, force: bool = False) -> bytes:
    if USE_DISK:
        path = CACHE_DIR / f"{key}.csv.gz"
        if force and path.exists():
            path.unlink()
        if not path.exists():
            print(f"  Downloading {key}.csv.gz ...", flush=True)
            r = requests.get(URLS[key], timeout=120, headers={"User-Agent": "MEP-Analyzer/4.0"})
            r.raise_for_status()
            path.write_bytes(r.content)
        else:
            print(f"  Cached {key}.csv.gz", flush=True)
        return path.read_bytes()
    else:
        print(f"  Downloading {key}.csv.gz ...", flush=True)
        r = requests.get(URLS[key], timeout=120, headers={"User-Agent": "MEP-Analyzer/4.0"})
        r.raise_for_status()
        return r.content


def load_data(force: bool = False) -> None:
    global _data
    CACHE_DIR.mkdir(exist_ok=True)
    print("\nLoading dataset ...", flush=True)

    # --- groups (tiny) ---
    groups = pd.read_csv(
        io.BytesIO(_get_raw("groups", force)),
        compression="gzip", usecols=["code", "short_label"]
    )
    group_map = dict(zip(groups["code"].astype(str), groups["short_label"].astype(str)))
    del groups

    # --- members (small) ---
    members = pd.read_csv(
        io.BytesIO(_get_raw("members", force)),
        compression="gzip", usecols=["id", "first_name", "last_name"]
    )
    members["id"] = pd.to_numeric(members["id"], errors="coerce").astype("Int32")
    members["full_name"] = (
        members["first_name"].fillna("").astype(str) + " " +
        members["last_name"].fillna("").astype(str)
    ).str.strip()
    del members["first_name"], members["last_name"]
    print(f"  {len(members):,} MEPs loaded", flush=True)

    # --- votes: keep only main votes ---
    votes_raw = _get_raw("votes", force)
    all_votes = pd.read_csv(
        io.BytesIO(votes_raw),
        compression="gzip",
        usecols=["id", "timestamp", "display_title", "description",
                 "amendment_subject", "amendment_number", "is_main",
                 "reference", "procedure_reference", "procedure_title",
                 "procedure_type", "procedure_stage", "result",
                 "count_for", "count_against", "count_abstention",
                 "count_did_not_vote"],
        low_memory=True
    )
    del votes_raw
    # Keep only main votes — this is the key memory reduction
    votes = all_votes[all_votes["is_main"] == True].copy()
    del all_votes
    votes["id"] = pd.to_numeric(votes["id"], errors="coerce").astype("Int32")
    print(f"  {len(votes):,} main votes loaded (amendments discarded)", flush=True)

    # --- member_votes: only rows for main vote IDs ---
    main_vote_ids = set(votes["id"].dropna().astype(int).tolist())
    mv_raw = _get_raw("member_votes", force)

    # Read in chunks and keep only rows matching main vote IDs
    chunks = []
    for chunk in pd.read_csv(
        io.BytesIO(mv_raw),
        compression="gzip",
        usecols=["vote_id", "member_id", "position", "group_code"],
        chunksize=200_000,
        low_memory=True
    ):
        chunk["vote_id"]   = pd.to_numeric(chunk["vote_id"],   errors="coerce").astype("Int32")
        chunk["member_id"] = pd.to_numeric(chunk["member_id"], errors="coerce").astype("Int32")
        filtered = chunk[chunk["vote_id"].isin(main_vote_ids)]
        if not filtered.empty:
            chunks.append(filtered)
    del mv_raw

    member_votes = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(
        columns=["vote_id", "member_id", "position", "group_code"]
    )
    # Categorical for low-cardinality columns
    member_votes["position"]   = member_votes["position"].astype("category")
    member_votes["group_code"] = member_votes["group_code"].astype("category")
    print(f"  {len(member_votes):,} individual votes kept (main votes only)", flush=True)

    _data = {
        "members":      members,
        "votes":        votes,
        "member_votes": member_votes,
        "group_map":    group_map,
    }
    print("  Dataset ready.\n", flush=True)


def safe(val):
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


@app.before_request
def _ensure_loaded():
    if not _data:
        try:
            load_data()
        except Exception as e:
            print(f"load_data error: {e}", flush=True)


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/search")
def search():
    mep_query   = request.args.get("mep",   "").strip()
    topic_query = request.args.get("topic", "").strip()
    refresh     = request.args.get("refresh", "0") == "1"

    if not mep_query or not topic_query:
        return jsonify({"error": "Both mep and topic parameters are required."}), 400

    if not _data or refresh:
        try:
            load_data(force=refresh)
        except Exception as e:
            return jsonify({"error": f"Failed to load dataset: {e}"}), 502

    members      = _data["members"]
    votes        = _data["votes"]
    member_votes = _data["member_votes"]
    group_map    = _data["group_map"]

    # 1. Find MEP
    mask = members["full_name"].str.lower().str.contains(mep_query.lower(), na=False)
    hits = members[mask]
    if hits.empty:
        return jsonify({"error": f"No MEP found matching '{mep_query}'. Try a partial last name."}), 404

    mep         = hits.iloc[0]
    mep_id      = mep["id"]
    mep_name    = str(mep["full_name"])
    all_matches = hits["full_name"].astype(str).tolist()

    # 2. Find main votes matching topic
    tq        = topic_query.lower()
    title_col = "display_title" if "display_title" in votes.columns else "title"

    masks = [votes[title_col].astype(str).str.lower().str.contains(tq, na=False)]
    for col in ("description", "procedure_title", "amendment_subject"):
        if col in votes.columns:
            masks.append(votes[col].astype(str).str.lower().str.contains(tq, na=False))
    combined = masks[0]
    for m in masks[1:]:
        combined = combined | m
    topic_votes = votes[combined].copy()

    if topic_votes.empty:
        return jsonify({"error": f"No main votes found for topic '{topic_query}'."}), 404

    # 3. Get this MEP's positions on those votes
    vote_ids = set(topic_votes["id"].tolist())
    mep_mv   = member_votes[
        (member_votes["member_id"] == mep_id) &
        (member_votes["vote_id"].isin(vote_ids))
    ].copy()

    if mep_mv.empty:
        return jsonify({"error": f"{mep_name} has no recorded votes on '{topic_query}'."}), 404

    # 4. Merge metadata
    keep_cols = ["id", title_col, "timestamp", "description", "amendment_subject",
                 "amendment_number", "is_main", "reference", "procedure_reference",
                 "procedure_title", "procedure_type", "procedure_stage", "result",
                 "count_for", "count_against", "count_abstention", "count_did_not_vote"]
    keep_cols = [c for c in keep_cols if c in topic_votes.columns]
    merged    = mep_mv.merge(topic_votes[keep_cols], left_on="vote_id", right_on="id", how="left")

    # 5. Build results
    counts = {"FOR": 0, "AGAINST": 0, "ABSTENTION": 0, "DID_NOT_VOTE": 0}
    result_votes = []

    for _, row in merged.iterrows():
        pos = str(row.get("position", "")).strip().upper()
        if pos not in counts:
            pos = "ABSTENTION"
        counts[pos] += 1

        gc          = safe(row.get("group_code"))
        group_label = group_map.get(str(gc), str(gc)) if gc else None
        cf   = to_int(row.get("count_for"))
        ca   = to_int(row.get("count_against"))
        cab  = to_int(row.get("count_abstention"))
        cdnv = to_int(row.get("count_did_not_vote"))

        result_votes.append({
            "vote_id":             int(row["vote_id"]),
            "date":                str(safe(row.get("timestamp")) or "")[:10],
            "timestamp":           safe(row.get("timestamp")),
            "title":               str(safe(row.get(title_col)) or "Untitled")[:200],
            "description":         safe(row.get("description")),
            "amendment_subject":   safe(row.get("amendment_subject")),
            "amendment_number":    safe(row.get("amendment_number")),
            "is_main":             True,
            "reference":           safe(row.get("reference")),
            "procedure_reference": safe(row.get("procedure_reference")),
            "procedure_title":     safe(row.get("procedure_title")),
            "procedure_type":      safe(row.get("procedure_type")),
            "procedure_stage":     safe(row.get("procedure_stage")),
            "result":              safe(row.get("result")),
            "position":            pos,
            "group":               group_label,
            "parliament_for":      cf,
            "parliament_against":  ca,
            "parliament_abstention": cab,
            "parliament_did_not_vote": cdnv,
            "parliament_total":    sum(x for x in [cf, ca, cab, cdnv] if x is not None) or None,
        })

    # Newest first
    result_votes.sort(key=lambda v: str(v.get("timestamp") or ""), reverse=True)

    total = len(result_votes)
    return jsonify({
        "mep_name":     mep_name,
        "mep_id":       int(mep_id),
        "topic":        topic_query,
        "total":        total,
        "main_total":   total,
        "counts":       counts,
        "all_counts":   counts,
        "percentages":  {k: round(v / total * 100, 1) for k, v in counts.items()},
        "pct_based_on": "main",
        "votes":        result_votes,
        "all_matches":  all_matches,
        "multiple":     len(all_matches) > 1,
    })


@app.route("/api/status")
def status():
    loaded = bool(_data)
    return jsonify({"loaded": loaded,
                    "members": len(_data["members"]) if loaded else 0,
                    "votes":   len(_data["votes"])   if loaded else 0})


if __name__ == "__main__":
    Path("static").mkdir(exist_ok=True)
    load_data()
    port  = int(os.environ.get("PORT", 5000))
    local = os.environ.get("RENDER") is None
    if local:
        import webbrowser, threading
        print(f"  Running at -> http://localhost:{port}")
        print("  Press Ctrl+C to stop\n")
        threading.Timer(1.2, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    app.run(debug=False, host="0.0.0.0", port=port)