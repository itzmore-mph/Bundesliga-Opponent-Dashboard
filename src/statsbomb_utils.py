# src/statsbomb_utils.py
import json
from pathlib import Path
import numpy as np
import pandas as pd

PITCH_X, PITCH_Y = 120.0, 80.0   # StatsBomb coords
GOAL_X, GOAL_Y = 120.0, 40.0
HALF_GOAL_WIDTH = 7.32 / 2.0


def _safe_xy(loc):
    if isinstance(loc, list) and len(loc) >= 2:
        return float(loc[0]), float(loc[1])
    return np.nan, np.nan


def _distance_angle(x, y):
    dist = np.hypot(GOAL_X - x, GOAL_Y - y)
    # simple angle proxy (opening angle to posts)
    angle = np.arctan2(HALF_GOAL_WIDTH, np.maximum(dist, 1e-6))
    return dist, angle


def load_matches(
    base_dir="data/statsbomb/data", competition_id=11, season_id=90
):
    """Return list of matches for a competition/season."""
    p = Path(base_dir) / "matches" / str(competition_id) / f"{season_id}.json"
    matches = json.loads(p.read_text(encoding="utf-8"))
    return matches


def iter_events_for_matches(base_dir, match_ids):
    """Yield (match_id, events_list)."""
    ev_dir = Path(base_dir) / "events"
    for mid in match_ids:
        ev = json.loads((ev_dir / f"{mid}.json").read_text(encoding="utf-8"))
        yield mid, ev


def load_season_shots(base_dir="data/statsbomb/data",
                      competition_id=11, season_id=90,
                      team_name=None, include_penalties=False):
    """
    Build a season-level shots DataFrame.
    If team_name is provided, includes only shots by that team
    (for all matches).
    Penalties excluded by default.
    """
    matches = load_matches(base_dir, competition_id, season_id)

    # Optional filter by exact team name at match level
    if team_name:
        matches = [
            m for m in matches
            if m["home_team"]["home_team_name"] == team_name
            or m["away_team"]["away_team_name"] == team_name
        ]

    rows = []
    ev_dir = Path(base_dir) / "events"

    for m in matches:
        mid = m["match_id"]
        home_team = m["home_team"]["home_team_name"]
        away_team = m["away_team"]["away_team_name"]

        # load events for this match
        ev_path = ev_dir / f"{mid}.json"
        if not ev_path.exists():
            # some open-data sets have missing matches; skip cleanly
            continue

        events = json.loads(ev_path.read_text(encoding="utf-8"))
        df = pd.json_normalize(events, sep=".")
        if df.empty:
            continue

        shots = df[df["type.name"] == "Shot"].copy()
        if shots.empty:
            continue

        # optional: drop penalties (guard if column missing)
        if not include_penalties:
            if "shot.type.name" in shots.columns:
                shots = shots[shots["shot.type.name"] != "Penalty"]

        # coords + basic features
        x, y = zip(*shots["location"].apply(_safe_xy))
        shots["x"], shots["y"] = x, y
        shots["distance"], shots["angle"] = _distance_angle(
            shots["x"], shots["y"]
        )
        shots["is_head"] = (
            shots.get(
                "shot.body_part.name", pd.Series(index=shots.index)
            ) == "Head"
        ).astype(int)
        shots["goal"] = (
            shots.get(
                "shot.outcome.name", pd.Series(index=shots.index)
            ) == "Goal"
        ).astype(int)

        # team labels from match metadata
        shots["home_team"] = home_team
        shots["away_team"] = away_team

        # shot team & opponent (compare to home/away strings)
        shots["shot_team"] = shots["team.name"]
        shots["opponent_team"] = np.where(
            shots["shot_team"] == home_team, away_team, home_team
        )

        shots["match_id"] = mid

        keep_cols = [
            "match_id", "shot_team", "opponent_team", "player.name",
            "x", "y", "distance", "angle", "is_head", "goal",
            "shot.type.name", "shot.technique.name", "minute", "second"
        ]
        # keep only columns that actually exist
        keep_cols = [c for c in keep_cols if c in shots.columns]
        rows.append(shots[keep_cols])

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out = out.dropna(subset=["x", "y"]) if not out.empty else out
    return out.reset_index(drop=True)
