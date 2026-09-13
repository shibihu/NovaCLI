"""Cache and incremental scanner for Project Intelligence 2.0."""

from __future__ import annotations

import json
from pathlib import Path

from nova.workspace.files import Workspace
from nova.intelligence.models import ProjectInfo
from nova.intelligence.scanner import ProjectScanner


class IntelligenceCache:
    """Stores and retrieves ProjectInfo from <project_root>/.nova/intelligence.json.

    Guarantees secret file contents and API keys are never written to disk.
    """

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    @property
    def cache_path(self) -> Path:
        return self.workspace.root / ".nova" / "intelligence.json"

    def load(self) -> dict | None:
        if not self.cache_path.exists():
            return None
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except (OSError, ValueError):
            pass
        return None

    def save(self, info: ProjectInfo) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        data = info.to_dict()
        self.cache_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def get_or_scan(self, *, force_refresh: bool = False) -> ProjectInfo:
        scanner = ProjectScanner(self.workspace)
        if not force_refresh:
            cached = self.load()
            if cached and isinstance(cached, dict):
                if cached.get("root") == str(self.workspace.root):
                    return ProjectInfo.from_dict(cached)
        info = scanner.scan()
        self.save(info)
        return info
