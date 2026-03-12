"""Per-skill usage and effectiveness tracking.

Tracks how often skills are viewed and whether sessions where a skill was
used tend to have successful outcomes. Scores stored in sidecar JSON file.
"""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List


SKILLS_DIR = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes")) / "skills"


class SkillMetrics:
    """Per-skill effectiveness tracking.

    File: SKILLS_DIR / ".metrics.json"
    Schema: { "skill-name": {"views": N, "score": 0.5, "last_used": "..."} }
    """
    EMA_ALPHA = 0.1

    def __init__(self, path: Path = None):
        self._path = path or SKILLS_DIR / ".metrics.json"
        self._data: Dict[str, dict] = {}

    def load(self):
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(self._data, ensure_ascii=False, indent=2)
        fd, tmp = tempfile.mkstemp(dir=str(self._path.parent), suffix=".tmp")
        try:
            os.write(fd, content.encode("utf-8"))
            os.fsync(fd)
            os.close(fd)
            os.replace(tmp, str(self._path))
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def record_view(self, skill_name: str):
        """Increment view count and update last_used timestamp."""
        entry = self._data.setdefault(skill_name, {"views": 0, "score": 0.5})
        entry["views"] = entry.get("views", 0) + 1
        entry["last_used"] = datetime.now().isoformat()

    def record_session_outcome(self, skills: List[str], tool_success_rate: float):
        """Update EMA scores for skills viewed this session."""
        for name in skills:
            entry = self._data.get(name)
            if not entry:
                continue
            old = entry.get("score", 0.5)
            entry["score"] = round(
                old * (1 - self.EMA_ALPHA) + tool_success_rate * self.EMA_ALPHA, 4
            )

    def get_label(self, skill_name: str) -> str:
        """Return qualitative effectiveness label for a skill."""
        entry = self._data.get(skill_name)
        if not entry or entry.get("views", 0) < 3:
            return "untested"
        score = entry.get("score", 0.5)
        views = entry.get("views", 0)
        if score >= 0.70 and views >= 5:
            return "reliable"
        elif score < 0.40 and views >= 5:
            return "unreliable"
        return "mixed" if views >= 5 else "untested"

    def get_metrics(self, names: List[str]) -> Dict[str, dict]:
        """Return metrics dict for display in tool responses."""
        result = {}
        for name in names:
            entry = self._data.get(name)
            if entry:
                result[name] = {
                    "views": entry.get("views", 0),
                    "label": self.get_label(name),
                    "last_used": entry.get("last_used"),
                }
        return result

    def get_flagged(self, threshold: float = 0.35) -> List[str]:
        """Return skill names with low effectiveness and sufficient usage."""
        return [
            name for name, d in self._data.items()
            if d.get("views", 0) >= 5 and d.get("score", 0.5) < threshold
        ]
