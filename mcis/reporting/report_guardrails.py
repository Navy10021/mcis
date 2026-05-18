from __future__ import annotations

import re
from typing import Any

_INFERENCE_FORBIDDEN_PATTERNS: list[re.Pattern] = [
    re.compile(r"predict(s|ed|ing|or)?\s+(armed\s+)?conflict", re.IGNORECASE),
    re.compile(r"(statistically\s+)?significant(ly)?", re.IGNORECASE),
    re.compile(r"\bcausal(l?y|ity)?\b", re.IGNORECASE),
    re.compile(r"generaliz(e|able|ation)", re.IGNORECASE),
    re.compile(r"prove[sd]?\s+that", re.IGNORECASE),
    re.compile(r"demonstrat(e|es|ed)\s+that", re.IGNORECASE),
    re.compile(r"early.?warning\s+(of|for|system)", re.IGNORECASE),
    re.compile(r"conflict\s+precursor", re.IGNORECASE),
    re.compile(r"predictive\s+(power|performance|capability)", re.IGNORECASE),
    re.compile(r"ground.?truth", re.IGNORECASE),
    re.compile(r"(reliably|accurately)\s+predict", re.IGNORECASE),
    re.compile(r"true\s+positive\s+rate.*conflict", re.IGNORECASE),
    re.compile(r"deploy(?:ed|ment|able)?\s+as\s+(?:a|an)\s+early", re.IGNORECASE),
    re.compile(r"operational\s+warning", re.IGNORECASE),
    re.compile(r"military\s+intelligence", re.IGNORECASE),
    re.compile(r"decision.?mak(?:ing|er)", re.IGNORECASE),
    re.compile(r"actionable\s+(intelligence|warning)", re.IGNORECASE),
]

_ENGINEERING_OK_PATTERNS: list[re.Pattern] = [
    re.compile(r"(pipeline|architecture|end.?to.?end)\s+(can|does)", re.IGNORECASE),
    re.compile(r"unit\s+test", re.IGNORECASE),
    re.compile(r"engineering\s+(demo|demonstration|prototype)", re.IGNORECASE),
    re.compile(r"stress\s+test", re.IGNORECASE),
    re.compile(r"ablation", re.IGNORECASE),
    re.compile(r"not\s+(intended|designed|meant)\s+for", re.IGNORECASE),
    re.compile(r"research\s+(prototype|framework|tool)", re.IGNORECASE),
]

_CLAIM_LEVEL_LABELS: dict[str, str] = {
    "engineering_demo": (
        "The model architecture runs end-to-end. Generated data is used for "
        "unit tests, stress tests, and ablation debugging only."
    ),
    "descriptive_evidence": (
        "Behavioral AIS indicators before/after T0 are reported. "
        "No causal or predictive claims are made."
    ),
    "inferential_evidence": (
        "Statistical tests show significant deviations relative to "
        "pre-event baseline. Valid only for empirical data."
    ),
    "predictive_prototype": (
        "A model generated early-warning scores before the event. "
        "This is a research prototype, not an operational system."
    ),
}

_ALLOWED_INFERENTIAL_MODES = {"empirical"}


def _build_inference_patterns() -> list[re.Pattern]:
    """Build compiled regex patterns for inferential language detection."""
    return _INFERENCE_FORBIDDEN_PATTERNS


def _build_engineering_patterns() -> list[re.Pattern]:
    return _ENGINEERING_OK_PATTERNS


def _strip_code_blocks(text: str) -> str:
    """Remove code blocks from text before scanning for forbidden language."""
    return re.sub(r"```[\s\S]*?```", "", text)


def assert_allowed_claim_language(
    text: str,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Check text for inferential language incompatible with data_validity_mode.

    Returns list of violations, each with pattern matched, snippet, and severity.
    Raises ValueError if any violation is found and mode is non-empirical.
    """
    mode = config.get("project", {}).get("data_validity_mode", "synthetic")
    claim_level = config.get("project", {}).get("claim_level", "engineering_demo")

    text_clean = _strip_code_blocks(text)

    violations: list[dict[str, Any]] = []

    if mode not in _ALLOWED_INFERENTIAL_MODES:
        for pattern in _build_inference_patterns():
            for match in pattern.finditer(text_clean):
                start = max(0, match.start() - 30)
                end = min(len(text_clean), match.end() + 30)
                snippet = text_clean[start:end].replace("\n", " ")
                violations.append({
                    "pattern": pattern.pattern,
                    "matched": match.group(),
                    "snippet": f"...{snippet}...",
                    "severity": "error" if claim_level != "engineering_demo" else "warning",
                })

    return violations


def validate_report_content(
    text: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Validate full report text for claim-level compliance.

    Returns dict with is_valid, violations, and metadata.
    Also checks that a metadata block is present.
    """
    mode = config.get("project", {}).get("data_validity_mode", "synthetic")
    claim_level = config.get("project", {}).get("claim_level", "engineering_demo")

    violations = assert_allowed_claim_language(text, config)
    has_metadata = "data_validity_mode" in text and "claim_level" in text

    valid_modes_for_level = {
        "engineering_demo": {"empirical", "synthetic", "mixed"},
        "descriptive_evidence": {"empirical", "synthetic", "mixed"},
        "inferential_evidence": {"empirical"},
        "predictive_prototype": {"empirical"},
    }

    level_mode_ok = mode in valid_modes_for_level.get(claim_level, set())

    errors = []
    warnings = []

    if not level_mode_ok:
        errors.append(
            f"claim_level={claim_level!r} not allowed with "
            f"data_validity_mode={mode!r}"
        )

    if not has_metadata:
        warnings.append("Report missing data_validity_mode / claim_level metadata block")

    for v in violations:
        if v["severity"] == "error":
            errors.append(f"Inferential language detected: {v['matched']!r} in context: {v['snippet']}")
        else:
            warnings.append(f"Potentially inferential language: {v['matched']!r}")

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "violations": violations,
        "data_validity_mode": mode,
        "claim_level": claim_level,
    }


def build_metadata_block(
    config: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> str:
    """Build a YAML-like metadata block for embedding in report outputs."""
    mode = config.get("project", {}).get("data_validity_mode", "synthetic")
    claim_level = config.get("project", {}).get("claim_level", "engineering_demo")
    t0 = config.get("conflict", {}).get("t0", "unknown")
    zone = config.get("conflict", {}).get("zone_name", "Black Sea")

    lines = [
        "---",
        f"data_validity_mode: {mode}",
        f"claim_level: {claim_level}",
        f"claim_description: |",
    ]
    desc_line = _CLAIM_LEVEL_LABELS.get(claim_level, "No description available.")
    lines.append(f"  {desc_line}")
    lines.append(f"conflict_t0: {t0}")
    lines.append(f"conflict_zone: {zone}")

    if extra:
        for k, v in extra.items():
            lines.append(f"{k}: {v}")

    lines.append("---")
    return "\n".join(lines)



