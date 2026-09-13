"""Project scanner and ecosystem detector for Project Intelligence 2.0."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Sequence

from nova.workspace.files import Workspace, detect_language
from nova.intelligence.models import (
    ProjectDependency,
    ProjectEntryPoint,
    ProjectGitInfo,
    ProjectInfo,
    ProjectTestInfo,
)

IGNORED_DIRS: frozenset[str] = frozenset({
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache",
    "dist", "build", ".idea", ".vscode", "target", "vendor", ".godot",
})

SECRET_NAMES: frozenset[str] = frozenset({
    ".env", ".netrc", ".npmrc", ".pypirc", ".git-credentials", "id_rsa", "id_ed25519",
})

SECRET_EXTENSIONS: frozenset[str] = frozenset({
    ".pem", ".key", ".p12", ".pfx", ".keystore", ".jks", ".crt",
})


def is_secret_file(path: Path) -> bool:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if name in SECRET_NAMES or name.startswith(".env.") or suffix in SECRET_EXTENSIONS:
        return True
    return False


class ProjectScanner:
    """Safely scans and extracts ProjectInfo without loading full files or secret contents."""

    def __init__(self, workspace: Workspace, *, max_files: int = 5_000) -> None:
        self.workspace = workspace
        self.max_files = max_files

    def scan(self) -> ProjectInfo:
        root = self.workspace.root
        files = self._collect_files()

        languages = self._detect_languages(files)
        ecosystems, frameworks, pkg_mgrs = self._detect_ecosystems_and_frameworks(files)
        dependencies = self._detect_dependencies(files)
        entry_points = self._detect_entry_points(files)
        important_files = self._detect_important_files(files)
        test_info = self._detect_tests(files)
        git_info = self._detect_git()

        total_bytes = sum(f.stat().st_size for f in files if f.exists())

        return ProjectInfo(
            name=root.name or "workspace",
            root=str(root),
            ecosystems=ecosystems,
            languages=languages,
            frameworks=frameworks,
            package_managers=pkg_mgrs,
            dependencies=dependencies,
            entry_points=entry_points,
            important_files=important_files,
            tests=test_info,
            git=git_info,
            total_files=len(files),
            total_bytes=total_bytes,
        )

    def _collect_files(self) -> list[Path]:
        collected: list[Path] = []
        root = self.workspace.root

        for current, dirs, files in os.walk(root):
            # Prune ignored directories
            dirs[:] = [d for d in dirs if d not in IGNORED_DIRS and not d.startswith(".")]

            for file in files:
                full_path = Path(current) / file
                if is_secret_file(full_path):
                    continue
                collected.append(full_path)
                if len(collected) >= self.max_files:
                    return collected
        return collected

    def _detect_languages(self, files: Sequence[Path]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for path in files:
            lang = detect_language(path)
            if lang:
                counts[lang] = counts.get(lang, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

    def _detect_ecosystems_and_frameworks(
        self, files: Sequence[Path]
    ) -> tuple[list[str], list[str], list[str]]:
        file_names = {f.name for f in files}
        rel_paths = {self.workspace.relative(f) for f in files}

        ecosystems: set[str] = set()
        frameworks: set[str] = set()
        package_managers: set[str] = set()

        # Python
        if any(f.endswith(".py") for f in rel_paths) or {"requirements.txt", "pyproject.toml", "setup.py"} & file_names:
            ecosystems.add("Python")
            if "requirements.txt" in file_names or "setup.py" in file_names:
                package_managers.add("pip")
            if "poetry.lock" in file_names:
                package_managers.add("poetry")
            if "uv.lock" in file_names:
                package_managers.add("uv")

        # Node / JS / TS
        if any(f.endswith((".js", ".ts", ".jsx", ".tsx")) for f in rel_paths) or "package.json" in file_names:
            ecosystems.add("Node.js")
            package_managers.add("npm")
            if "yarn.lock" in file_names:
                package_managers.add("yarn")
            if "pnpm-lock.yaml" in file_names:
                package_managers.add("pnpm")

        # Luau / Roblox
        if any(f.endswith((".lua", ".luau")) for f in rel_paths) or {"default.project.json", "wally.toml"} & file_names:
            ecosystems.add("Roblox/Luau")
            if "wally.toml" in file_names:
                package_managers.add("wally")

        # Godot
        if "project.godot" in file_names or any(f.endswith(".gd") for f in rel_paths):
            ecosystems.add("Godot")

        # Web
        if any(f.endswith((".html", ".css")) for f in rel_paths):
            ecosystems.add("Web (HTML/CSS)")

        # Framework detection via imports/deps inspection (lightweight)
        deps = self._detect_dependencies(files)
        dep_names = {d.name.lower() for d in deps}

        for fw in ("fastapi", "flask", "django", "pydantic", "groq", "react", "vue", "vite", "express", "next"):
            if fw in dep_names:
                frameworks.add(fw)

        return sorted(ecosystems), sorted(frameworks), sorted(package_managers)

    def _detect_dependencies(self, files: Sequence[Path]) -> list[ProjectDependency]:
        deps: list[ProjectDependency] = []

        # Python: requirements.txt
        for path in files:
            if path.name == "requirements.txt":
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                    for line in text.splitlines():
                        line = line.strip()
                        if line and not line.startswith("#"):
                            pkg = line.split("==")[0].split(">=")[0].split("<=")[0].strip()
                            if pkg:
                                deps.append(ProjectDependency(name=pkg, ecosystem="Python"))
                except OSError:
                    pass

            # Node: package.json
            elif path.name == "package.json":
                try:
                    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
                    if isinstance(data, dict):
                        d_deps = data.get("dependencies", {})
                        if isinstance(d_deps, dict):
                            for k, v in d_deps.items():
                                deps.append(ProjectDependency(name=str(k), version=str(v), dev=False, ecosystem="Node.js"))
                        dev_deps = data.get("devDependencies", {})
                        if isinstance(dev_deps, dict):
                            for k, v in dev_deps.items():
                                deps.append(ProjectDependency(name=str(k), version=str(v), dev=True, ecosystem="Node.js"))
                except (OSError, ValueError):
                    pass

        return deps[:100]

    def _detect_entry_points(self, files: Sequence[Path]) -> list[ProjectEntryPoint]:
        entry_points: list[ProjectEntryPoint] = []
        rel_paths = [self.workspace.relative(f) for f in files]

        candidates = (
            "main.py", "app.py", "nova.py", "server.py", "__main__.py",
            "index.js", "main.js", "app.js", "server.js",
            "index.ts", "main.ts", "app.ts", "server.ts",
            "project.godot",
        )

        for rel in sorted(rel_paths, key=lambda p: len(Path(p).parts)):
            p_name = Path(rel).name
            if p_name in candidates or rel in candidates:
                entry_points.append(ProjectEntryPoint(path=rel, type="main"))
            if len(entry_points) >= 10:
                break

        return entry_points

    def _detect_important_files(self, files: Sequence[Path]) -> list[str]:
        important: list[str] = []
        for path in files:
            rel = self.workspace.relative(path)
            parts = Path(rel).parts
            if len(parts) <= 2 and path.suffix in {".py", ".ts", ".js", ".json", ".toml", ".md", ".godot"}:
                important.append(rel)
            if len(important) >= 20:
                break
        return important

    def _detect_tests(self, files: Sequence[Path]) -> ProjectTestInfo:
        frameworks: set[str] = set()
        test_files: list[str] = []

        for path in files:
            rel = self.workspace.relative(path)
            name = path.name
            parts = set(Path(rel).parts)

            is_test = (
                bool(parts & {"tests", "test", "spec", "__tests__"})
                or name.startswith("test_")
                or name.endswith("_test.py")
                or name.endswith((".test.js", ".test.ts", ".spec.js", ".spec.ts"))
            )
            if is_test:
                test_files.append(rel)
                if name.endswith(".py"):
                    frameworks.add("pytest")
                elif name.endswith((".js", ".ts")):
                    frameworks.add("vitest/jest")

        return ProjectTestInfo(
            frameworks=sorted(frameworks),
            test_files=test_files[:50],
            total_tests=len(test_files),
        )

    def _detect_git(self) -> ProjectGitInfo:
        git_dir = self.workspace.root / ".git"
        if not git_dir.exists():
            return ProjectGitInfo(has_git=False)

        try:
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.workspace.root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            branch = None

        modified, staged, untracked = [], [], []
        try:
            status_out = subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=self.workspace.root,
                text=True,
                stderr=subprocess.DEVNULL,
            )
            for line in status_out.splitlines():
                if not line or len(line) < 3:
                    continue
                index_status = line[0]
                work_status = line[1]
                file_path = line[3:].strip()

                if index_status in ("M", "A", "R"):
                    staged.append(file_path)
                if work_status == "M":
                    modified.append(file_path)
                if index_status == "?" and work_status == "?":
                    untracked.append(file_path)
        except Exception:
            pass

        recent_commits = []
        try:
            log_out = subprocess.check_output(
                ["git", "log", "-n", "3", "--oneline"],
                cwd=self.workspace.root,
                text=True,
                stderr=subprocess.DEVNULL,
            )
            recent_commits = [line.strip() for line in log_out.splitlines() if line.strip()]
        except Exception:
            pass

        return ProjectGitInfo(
            has_git=True,
            branch=branch,
            modified_files=modified[:20],
            staged_files=staged[:20],
            untracked_files=untracked[:20],
            recent_commits=recent_commits,
        )
