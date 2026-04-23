from __future__ import annotations

from scripts import precommit_plan


def test_docs_only_paths_use_fast_path() -> None:
    plan = precommit_plan.classify_paths(
        [
            "README.md",
            "docs/decisions/0043-public-api-physical-plan-coverage.md",
            "docs/ops/intake-index.json",
        ]
    )

    assert plan.scope == precommit_plan.DOCS_ONLY_SCOPE


def test_docs_assets_use_fast_path() -> None:
    plan = precommit_plan.classify_paths(["docs/images/architecture.png"])

    assert plan.scope == precommit_plan.DOCS_ONLY_SCOPE


def test_code_paths_require_full_checks() -> None:
    plan = precommit_plan.classify_paths(["README.md", "src/vibespatial/runtime/__init__.py"])

    assert plan.scope == precommit_plan.FULL_SCOPE
    assert plan.reason == "non-doc staged path: src/vibespatial/runtime/__init__.py"


def test_requirements_txt_requires_full_checks() -> None:
    plan = precommit_plan.classify_paths(["requirements.txt"])

    assert plan.scope == precommit_plan.FULL_SCOPE


def test_empty_path_set_requires_full_checks() -> None:
    plan = precommit_plan.classify_paths([])

    assert plan.scope == precommit_plan.FULL_SCOPE


def test_force_full_env_var_overrides_docs_only(monkeypatch) -> None:
    monkeypatch.setenv("VIBESPATIAL_PRECOMMIT_FORCE_FULL", "1")

    plan = precommit_plan.classify_paths(["README.md"])

    assert plan.scope == precommit_plan.FULL_SCOPE
    assert plan.reason == "forced by VIBESPATIAL_PRECOMMIT_FORCE_FULL=1"
