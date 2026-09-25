"""Stage E: reviewer decisions logged against the model's output (SQLite)."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .schema import Finding, Level

SCHEMA = """
CREATE TABLE IF NOT EXISTS reviews (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  finding_id TEXT NOT NULL,
  action TEXT NOT NULL,
  prior_level TEXT,
  new_level TEXT,
  reviewer TEXT,
  reviewed_at TEXT NOT NULL
);
"""


MIGRATIONS = {"blind": "INTEGER DEFAULT 0", "model_level_shown": "INTEGER DEFAULT 1"}


class ReviewLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.executescript(SCHEMA)
        self.migrate()

    def migrate(self) -> List[str]:
        """Add the blind-QC columns to a reviews table created before they existed; returns the columns added."""
        have = {row[1] for row in self.conn.execute("PRAGMA table_info(reviews)").fetchall()}
        added = []
        for col, decl in MIGRATIONS.items():
            if col not in have:
                self.conn.execute(f"ALTER TABLE reviews ADD COLUMN {col} {decl}")
                added.append(col)
        self.conn.commit()
        return added

    def record_blind(self, run_id: str, f: Finding, level: Level, reviewer: str) -> dict:
        """Loop 3 blind QC: the reviewer graded without seeing the model's level. Stored with blind=1 and the model's
        level as prior_level; the finding itself is not mutated and the row is excluded from timeline/agreement.
        One blind row per (run, finding): a second call returns the stored row with `existing: True` and inserts
        nothing, so one finding can never supply several blind pairs."""
        prior = self.conn.execute("SELECT prior_level, new_level, reviewer, reviewed_at FROM reviews WHERE run_id=? AND finding_id=? AND blind=1 ORDER BY id LIMIT 1", (run_id, f.finding_id)).fetchone()
        if prior:
            return {"run_id": run_id, "finding_id": f.finding_id, "action": "blind", "prior_level": prior[0], "new_level": prior[1], "reviewer": prior[2], "reviewed_at": prior[3], "blind": 1, "existing": True}
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.conn.execute(
            "INSERT INTO reviews(run_id, finding_id, action, prior_level, new_level, reviewer, reviewed_at, blind, model_level_shown) VALUES (?,?,?,?,?,?,?,1,0)",
            (run_id, f.finding_id, "blind", f.unified.level, level, reviewer, now),
        )
        self.conn.commit()
        return {"run_id": run_id, "finding_id": f.finding_id, "action": "blind", "prior_level": f.unified.level, "new_level": level, "reviewer": reviewer, "reviewed_at": now, "blind": 1, "existing": False}

    def blind_ids(self, run_id: Optional[str] = None) -> set:
        """finding_ids that already hold a blind row (excluded from drift.blind_sample)."""
        q = "SELECT DISTINCT finding_id FROM reviews WHERE blind=1" + (" AND run_id=?" if run_id else "")
        return {row[0] for row in self.conn.execute(q, (run_id,) if run_id else ()).fetchall()}

    def decisions(self, run_id: Optional[str] = None) -> List[dict]:
        """Every row (sighted and blind) as dicts in insertion order, so drift.review_health embeds no SQL."""
        q = "SELECT run_id, finding_id, action, prior_level, new_level, reviewer, reviewed_at, blind, model_level_shown FROM reviews" + (" WHERE run_id=?" if run_id else "") + " ORDER BY id"
        cols = ["run_id", "finding_id", "action", "prior_level", "new_level", "reviewer", "reviewed_at", "blind", "model_level_shown"]
        return [dict(zip(cols, row)) for row in self.conn.execute(q, (run_id,) if run_id else ()).fetchall()]

    def apply(self, run_id: str, f: Finding, action: str, reviewer: str, new_level: Optional[Level] = None) -> Finding:
        assert action in ("accepted", "overridden", "marked_u")
        prior = f.unified.level
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if action == "overridden":
            assert new_level is not None
            f.unified.level = new_level
        elif action == "marked_u":
            f.unified.level = "U"
            new_level = "U"
        else:
            new_level = prior
        f.review.status = action
        f.review.reviewer = reviewer
        f.review.reviewed_at = now
        f.review.prior_level = prior
        self.conn.execute(
            "INSERT INTO reviews(run_id, finding_id, action, prior_level, new_level, reviewer, reviewed_at) VALUES (?,?,?,?,?,?,?)",
            (run_id, f.finding_id, action, prior, new_level, reviewer, now),
        )
        self.conn.commit()
        return f

    def timeline(self, run_id: Optional[str] = None) -> List[dict]:
        """FR-19: one row per decision in order, with the running agreement rate (accepted / decisions so far)."""
        q = "SELECT reviewed_at, finding_id, action, prior_level, new_level, reviewer FROM reviews WHERE action != 'blind'" + (" AND run_id=?" if run_id else "") + " ORDER BY id"
        rows = self.conn.execute(q, (run_id,) if run_id else ()).fetchall()
        out, acc = [], 0
        for n, (ts, fid, action, prior, new, reviewer) in enumerate(rows, start=1):
            acc += int(action == "accepted")
            out.append({"n": n, "reviewed_at": ts, "finding_id": fid, "action": action, "prior_level": prior, "new_level": new, "reviewer": reviewer, "agreement_rate": acc / n})
        return out

    def agreement(self, run_id: Optional[str] = None) -> dict:
        q = "SELECT action, COUNT(*) FROM reviews WHERE action != 'blind'" + (" AND run_id=?" if run_id else "") + " GROUP BY action"
        rows = self.conn.execute(q, (run_id,) if run_id else ()).fetchall()
        counts = {a: n for a, n in rows}
        total = sum(counts.values())
        return {"total": total, "accepted": counts.get("accepted", 0), "overridden": counts.get("overridden", 0), "marked_u": counts.get("marked_u", 0), "agreement_rate": (counts.get("accepted", 0) / total) if total else None}
