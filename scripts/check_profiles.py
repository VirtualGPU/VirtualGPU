"""Validate every GPU profile in profiles/ against the schema and its provenance rules."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from virtualgpu.profile import ProfileValidationError, load_profile  # noqa: E402

LEAF_KEYS = {"value", "rel_uncertainty", "unit", "method"}


def leaf_sources(obj, path="measured"):
    if isinstance(obj, dict):
        if LEAF_KEYS.issubset(obj):
            return [(path, obj.get("source"))]
        return [leaf for k, v in obj.items() for leaf in leaf_sources(v, f"{path}.{k}")]
    if isinstance(obj, list):
        return [leaf for i, v in enumerate(obj) for leaf in leaf_sources(v, f"{path}[{i}]")]
    return []


def check(path):
    try:
        profile = load_profile(path, schema_path=ROOT / "schemas" / "gpu_profile.schema.json")
    except (OSError, ValueError, ProfileValidationError) as exc:
        return [str(exc)], None

    errors = []
    prov = profile["provenance"]
    source = prov["source"]
    if source not in ("measured", "partial", "spec_derived"):
        return [f"invalid provenance.source={source!r}"], None

    required = ["profile_revision", "calibration_run_id", "timestamp_utc"]
    if source in ("measured", "partial"):
        required += ["driver_version", "toolkit_version", "suite_version", "host_os"]
    for field in required:
        value = prov.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"provenance.{field} must be a non-empty string for source={source!r}")

    leaves = leaf_sources(profile["measured"])
    if not leaves:
        return errors + ["no measured values found"], None
    bad = [(p, s) for p, s in leaves if s not in ("measured", "estimated")]
    if bad:
        sample = ", ".join(f"{p}={s!r}" for p, s in bad[:5])
        return errors + [f"invalid or missing leaf source: {sample}"], None

    measured = sum(1 for _, s in leaves if s == "measured")
    estimated = len(leaves) - measured
    if source == "measured" and estimated:
        errors.append(f"source is 'measured' but {estimated} values are 'estimated'")
    elif source == "spec_derived" and measured:
        errors.append(f"source is 'spec_derived' but {measured} values are 'measured'")
    elif source == "partial" and not (measured and estimated):
        errors.append(f"source is 'partial' but found {measured} measured, {estimated} estimated")

    ident = profile["identity"]
    summary = (
        f"{ident['sku']}, {ident['arch']}, {source}, {measured} measured, {estimated} estimated"
    )
    return errors, summary


def main() -> int:
    paths = sorted((ROOT / "profiles").glob("*.json"))
    if not paths:
        print("No profiles found in profiles/", file=sys.stderr)
        return 1
    failed = False
    for path in paths:
        errors, summary = check(path)
        name = path.relative_to(ROOT).as_posix()
        if errors:
            failed = True
            for err in errors:
                print(f"{name}: {err}", file=sys.stderr)
        else:
            print(f"OK {name} ({summary})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
