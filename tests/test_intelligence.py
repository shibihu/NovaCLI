"""Tests for Project Intelligence 2.0 subsystem."""

from __future__ import annotations

from pathlib import Path
from nova.workspace.files import Workspace
from nova.intelligence.models import ProjectInfo
from nova.intelligence.scanner import ProjectScanner, is_secret_file
from nova.intelligence.cache import IntelligenceCache


def test_secret_file_detection() -> None:
    assert is_secret_file(Path(".env")) is True
    assert is_secret_file(Path(".env.production")) is True
    assert is_secret_file(Path("id_rsa")) is True
    assert is_secret_file(Path("server.key")) is True
    assert is_secret_file(Path("cert.pem")) is True
    assert is_secret_file(Path("main.py")) is False


def test_project_scanner_python(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("print('hello')", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("fastapi==0.115.0\npydantic\n", encoding="utf-8")
    (tmp_path / "test_main.py").write_text("def test_ok(): pass", encoding="utf-8")
    (tmp_path / ".env").write_text("GROQ_API_KEY=gsk_secret", encoding="utf-8")

    workspace = Workspace(tmp_path)
    scanner = ProjectScanner(workspace)
    info = scanner.scan()

    assert "Python" in info.languages or "Python" in info.ecosystems
    assert "pip" in info.package_managers
    assert any(d.name == "fastapi" for d in info.dependencies)
    assert any(ep.path == "main.py" for ep in info.entry_points)
    assert "pytest" in info.tests.frameworks or info.tests.total_tests >= 1
    # Check that secrets are excluded
    collected_names = [f.name for f in scanner._collect_files()]
    assert ".env" not in collected_names
    assert "gsk_secret" not in str(info.to_dict())


def test_project_scanner_node(tmp_path: Path) -> None:
    package_json = {
        "name": "my-app",
        "dependencies": {"react": "^18.0.0"},
        "devDependencies": {"jest": "^29.0.0"},
    }
    (tmp_path / "package.json").write_text(str(package_json).replace("'", '"'), encoding="utf-8")
    (tmp_path / "index.js").write_text("console.log('hi')", encoding="utf-8")

    workspace = Workspace(tmp_path)
    scanner = ProjectScanner(workspace)
    info = scanner.scan()

    assert "Node.js" in info.ecosystems
    assert "npm" in info.package_managers
    assert any(d.name == "react" for d in info.dependencies)
    assert any(ep.path == "index.js" for ep in info.entry_points)


def test_intelligence_cache(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("import fastapi", encoding="utf-8")
    workspace = Workspace(tmp_path)
    cache = IntelligenceCache(workspace)

    info = cache.get_or_scan()
    assert cache.cache_path.exists()

    cached_data = cache.load()
    assert cached_data is not None
    assert cached_data["name"] == info.name
    assert "GROQ_API_KEY" not in str(cached_data)
