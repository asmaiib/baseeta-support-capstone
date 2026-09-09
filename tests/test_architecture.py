"""Architecture rules, enforced by a test rather than by hoping.

Three claims this course makes about the codebase, each of which is only true for
as long as something checks it:

1. application code never imports a provider SDK;
2. prompt text lives in the registry, never inline in Python;
3. no model call is unbounded — ``max_tokens`` is always set.

Each of these is a grep in CI in the instructor package. Here they are tests, so
they fail on a laptop before they fail on a pull request.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src" / "retail_support"
ALLOWED_SDK_IMPORTERS = {"openai_compat.py", "anthropic_client.py"}


def python_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def test_no_provider_sdk_outside_the_adapters():
    offenders: list[str] = []
    for path in python_files(SRC):
        if path.name in ALLOWED_SDK_IMPORTERS:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for name in names:
                if name.split(".")[0] in {"openai", "anthropic"}:
                    offenders.append(f"{path.relative_to(SRC)}:{node.lineno} imports {name}")
    assert offenders == [], (
        "provider SDKs belong behind the boundary. Every one of these turns a "
        "config change into a rewrite:\n  " + "\n  ".join(offenders)
    )


def test_no_inline_prompt_text_in_code():
    """`rg 'You are ' src/ --type py` must find nothing outside the registry."""
    pattern = re.compile(r'"(?:[^"\n]*\b(?:You are|أنت مساعد)\b[^"\n]*)"')
    offenders = []
    for path in python_files(SRC):
        if path.parts[-2:] == ("prompts", "registry.py"):
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if line.lstrip().startswith("#"):
                continue
            if pattern.search(line) and "prompt" not in line.lower():
                offenders.append(f"{path.relative_to(SRC)}:{number}: {line.strip()[:70]}")
    assert offenders == [], (
        "prompt text lives in src/retail_support/prompts/library as a versioned file:\n  "
        + "\n  ".join(offenders)
    )


def test_every_llm_request_bounds_max_tokens():
    offenders = []
    for path in python_files(SRC):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name != "LLMRequest":
                continue
            keywords = {kw.arg for kw in node.keywords}
            if "max_tokens" not in keywords:
                offenders.append(f"{path.relative_to(SRC)}:{node.lineno}")
    assert offenders == [], (
        "a request without max_tokens is a demo default that becomes a production "
        "incident:\n  " + "\n  ".join(offenders)
    )


def test_every_prompt_file_has_front_matter_and_a_changelog():
    from retail_support.prompts.registry import list_prompts, load_prompt

    prompts = list_prompts()
    assert prompts, "the registry is empty"
    for prompt_id, versions in prompts.items():
        for version in versions:
            artifact = load_prompt(f"{prompt_id}.{version}")
            assert artifact.changelog, f"{prompt_id}.{version} ships without a changelog line"
            assert artifact.text.strip()


def test_rendering_without_a_required_variable_fails_loudly():
    from retail_support.prompts.registry import MissingPromptVariable, load_prompt

    prompt = load_prompt("answer_faq.v1")
    with pytest.raises(MissingPromptVariable):
        prompt.render()


def test_the_canary_is_planted_in_every_rendered_system_prompt():
    from retail_support.prompts.registry import CANARY, load_prompt

    rendered = load_prompt("answer_faq.v1").render(product_catalog="x")
    assert CANARY in rendered
