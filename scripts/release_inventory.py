"""Release candidate file selection and non-destructive hygiene checks."""
from pathlib import Path
import hashlib
import re

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE = {
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "outputs", "build", "dist", "wandb", "logs", "runs", "checkpoints",
    "private", "data", "datasets", ".venv",
}
ROOT_FILES = {
    ".gitignore", ".gitattributes", "README.md", "LICENSE", "NOTICE", "CITATION.cff",
    "CHANGELOG.md", "RELEASE_NOTES.md", "RELEASE_MANIFEST.json",
    "setup.cfg", "pyproject.toml", "MANIFEST.in", "requirements.txt", "requirements-train.txt",
}
SOURCE_DIRS = {"cervithink", "scripts", "tests", "docs", "configs"}
SUFFIXES = {".py", ".sh", ".md", ".json", ".jsonl", ".example"}
LICENSE_HASH = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"


def release_files():
    selected = []
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT)
        if any(part in EXCLUDE or part.endswith(".egg-info") for part in rel.parts):
            continue
        if path.is_file():
            selected.append(path)
    return sorted(selected)


def check_inventory():
    issues = []
    secrets = [
        re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
        re.compile(r"hf_[A-Za-z0-9]{20,}"),
        re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
        re.compile(r"AKIA[0-9A-Z]{16}"),
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    ]
    local_paths = [
        re.compile(r"[A-Za-z]:[\\/](?:poster|Users[\\/])", re.I),
        re.compile(r"/(?:home|Users)/[A-Za-z0-9_.-]+/"),
    ]
    for path in release_files():
        rel = path.relative_to(ROOT)
        if path.is_symlink():
            issues.append(f"Symlink is not a release source file: {rel}")
            continue
        allowed = (
            (len(rel.parts) == 1 and rel.name in ROOT_FILES)
            or (rel.parts[0] in SOURCE_DIRS and path.suffix in SUFFIXES)
        )
        if not allowed:
            issues.append(f"Unexpected release file: {rel}")
            continue
        if path.stat().st_size > 2_000_000:
            issues.append(f"Unexpected large source file: {rel}")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            issues.append(f"Non-UTF8/binary content: {rel}")
            continue
        for pattern in secrets + local_paths:
            if pattern.search(text):
                # Only filename is printed; matched secret contents are not exposed.
                issues.append(f"Review sensitive/local-only pattern: {rel}")
                break
    if hashlib.sha256((ROOT/"LICENSE").read_bytes().replace(b"\r\n", b"\n")).hexdigest() != LICENSE_HASH:
        issues.append("LICENSE differs from the standard Apache-2.0 text")
    return issues
