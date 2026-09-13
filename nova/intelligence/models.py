"""Project Intelligence 2.0 data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProjectFile:
    path: str
    size: int = 0
    extension: str = ""
    language: str | None = None
    is_important: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "size": self.size,
            "extension": self.extension,
            "language": self.language,
            "is_important": self.is_important,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectFile:
        return cls(
            path=str(data.get("path", "")),
            size=int(data.get("size") or 0),
            extension=str(data.get("extension", "")),
            language=data.get("language"),
            is_important=bool(data.get("is_important")),
        )


@dataclass
class ProjectDependency:
    name: str
    version: str | None = None
    dev: bool = False
    ecosystem: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "dev": self.dev,
            "ecosystem": self.ecosystem,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectDependency:
        return cls(
            name=str(data.get("name", "")),
            version=data.get("version"),
            dev=bool(data.get("dev")),
            ecosystem=str(data.get("ecosystem", "")),
        )


@dataclass
class ProjectEntryPoint:
    path: str
    type: str = "main"

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "type": self.type}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectEntryPoint:
        return cls(
            path=str(data.get("path", "")),
            type=str(data.get("type", "main")),
        )


@dataclass
class ProjectTestInfo:
    frameworks: list[str] = field(default_factory=list)
    test_files: list[str] = field(default_factory=list)
    total_tests: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "frameworks": self.frameworks,
            "test_files": self.test_files,
            "total_tests": self.total_tests,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectTestInfo:
        return cls(
            frameworks=list(data.get("frameworks") or []),
            test_files=list(data.get("test_files") or []),
            total_tests=int(data.get("total_tests") or 0),
        )


@dataclass
class ProjectGitInfo:
    has_git: bool = False
    branch: str | None = None
    modified_files: list[str] = field(default_factory=list)
    staged_files: list[str] = field(default_factory=list)
    untracked_files: list[str] = field(default_factory=list)
    recent_commits: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_git": self.has_git,
            "branch": self.branch,
            "modified_files": self.modified_files,
            "staged_files": self.staged_files,
            "untracked_files": self.untracked_files,
            "recent_commits": self.recent_commits,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectGitInfo:
        return cls(
            has_git=bool(data.get("has_git")),
            branch=data.get("branch"),
            modified_files=list(data.get("modified_files") or []),
            staged_files=list(data.get("staged_files") or []),
            untracked_files=list(data.get("untracked_files") or []),
            recent_commits=list(data.get("recent_commits") or []),
        )


@dataclass
class ProjectInfo:
    name: str
    root: str
    ecosystems: list[str] = field(default_factory=list)
    languages: dict[str, int] = field(default_factory=dict)
    frameworks: list[str] = field(default_factory=list)
    package_managers: list[str] = field(default_factory=list)
    dependencies: list[ProjectDependency] = field(default_factory=list)
    entry_points: list[ProjectEntryPoint] = field(default_factory=list)
    important_files: list[str] = field(default_factory=list)
    tests: ProjectTestInfo = field(default_factory=ProjectTestInfo)
    git: ProjectGitInfo = field(default_factory=ProjectGitInfo)
    total_files: int = 0
    total_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "root": self.root,
            "ecosystems": self.ecosystems,
            "languages": self.languages,
            "frameworks": self.frameworks,
            "package_managers": self.package_managers,
            "dependencies": [d.to_dict() for d in self.dependencies],
            "entry_points": [e.to_dict() for e in self.entry_points],
            "important_files": self.important_files,
            "tests": self.tests.to_dict(),
            "git": self.git.to_dict(),
            "total_files": self.total_files,
            "total_bytes": self.total_bytes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectInfo:
        return cls(
            name=str(data.get("name", "")),
            root=str(data.get("root", "")),
            ecosystems=list(data.get("ecosystems") or []),
            languages=dict(data.get("languages") or {}),
            frameworks=list(data.get("frameworks") or []),
            package_managers=list(data.get("package_managers") or []),
            dependencies=[
                ProjectDependency.from_dict(d)
                for d in (data.get("dependencies") or [])
                if isinstance(d, dict)
            ],
            entry_points=[
                ProjectEntryPoint.from_dict(e)
                for e in (data.get("entry_points") or [])
                if isinstance(e, dict)
            ],
            important_files=list(data.get("important_files") or []),
            tests=ProjectTestInfo.from_dict(dict(data.get("tests") or {})),
            git=ProjectGitInfo.from_dict(dict(data.get("git") or {})),
            total_files=int(data.get("total_files") or 0),
            total_bytes=int(data.get("total_bytes") or 0),
        )
