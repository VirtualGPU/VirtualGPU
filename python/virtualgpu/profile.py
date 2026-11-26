"""GPU profile schema validation and loading utilities."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "gpu_profile.schema.json"


def _profile_search_dirs() -> list[Path]:
    """Directories holding bundled GPU profiles: inside the installed package,
    or the repository's profiles/ in a source checkout."""
    pkg_dir = Path(__file__).resolve().parent
    candidates = [pkg_dir / "profiles", pkg_dir.parents[1] / "profiles"]
    return [d for d in candidates if d.is_dir()]


def list_profiles() -> list[str]:
    """Return sorted names (without .json) of bundled GPU profiles."""
    names: set[str] = set()
    for d in _profile_search_dirs():
        for p in d.glob("*.json"):
            if p.is_file():
                names.add(p.stem)
    return sorted(names)


def profile_path(name: str) -> Path:
    """Resolve a bundled profile name (e.g. 'rtx4070' or 'gb10') to its JSON path."""
    stem = name[:-5] if name.endswith(".json") else name
    for d in _profile_search_dirs():
        candidate = d / f"{stem}.json"
        if candidate.is_file():
            return candidate
    available = list_profiles()
    hint = f" Available bundled profiles: {', '.join(available)}" if available else ""
    raise FileNotFoundError(f"Bundled GPU profile not found: {name!r}.{hint}")


class ProfileValidationError(ValueError):
    """Raised when a GPU profile fails schema or semantic validation."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        summary = "\n".join(f"- {err}" for err in errors)
        super().__init__(f"GPU profile validation failed ({len(errors)} error(s)):\n{summary}")


def load_schema(schema_path: Path | str | None = None) -> dict[str, Any]:
    if schema_path is None:
        try:
            from virtualgpu import _core

            return json.loads(_core.schema("gpu_profile"))
        except Exception:
            schema_path = DEFAULT_SCHEMA_PATH
    path = Path(schema_path)
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _check_measured_value(mv: dict[str, Any], path: str, errors: list[str]) -> None:
    lo = mv.get("ci95_low")
    hi = mv.get("ci95_high")
    val = mv["value"]
    if lo is not None and hi is not None:
        if lo > hi:
            errors.append(f"{path}: ci95_low ({lo}) > ci95_high ({hi})")
        if not (lo <= val <= hi):
            errors.append(f"{path}: value ({val}) outside [ci95_low={lo}, ci95_high={hi}]")


def _walk_measured_values(obj: Any, path: str, errors: list[str]) -> None:
    if isinstance(obj, dict):
        if {"value", "rel_uncertainty", "unit", "method"}.issubset(obj.keys()):
            _check_measured_value(obj, path, errors)
            return
        for key, val in obj.items():
            _walk_measured_values(val, f"{path}.{key}", errors)
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            _walk_measured_values(item, f"{path}[{idx}]", errors)


def _check_monotonic_curve(
    points: list[dict[str, Any]],
    x_key: str,
    max_x: int,
    path: str,
    errors: list[str],
) -> None:
    prev = 0
    for idx, pt in enumerate(points):
        x = pt[x_key]
        if x <= prev:
            errors.append(f"{path}[{idx}].{x_key} ({x}) must be strictly increasing (prev={prev})")
        if x > max_x:
            errors.append(f"{path}[{idx}].{x_key} ({x}) exceeds hardware limit ({max_x})")
        prev = x


def _semantic_checks(profile: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    specs = profile["specs"]
    measured = profile["measured"]

    _walk_measured_values(measured, "measured", errors)

    sm_count = specs["sm_count"]
    mem = measured["memory"]
    _check_monotonic_curve(
        mem["dram_bw_by_active_sms"],
        "active_sms",
        sm_count,
        "measured.memory.dram_bw_by_active_sms",
        errors,
    )
    _check_monotonic_curve(
        mem["l2_bw_by_active_sms"],
        "active_sms",
        sm_count,
        "measured.memory.l2_bw_by_active_sms",
        errors,
    )
    if "l2_bw_by_active_sms_cp_async" in mem:
        _check_monotonic_curve(
            mem["l2_bw_by_active_sms_cp_async"],
            "active_sms",
            sm_count,
            "measured.memory.l2_bw_by_active_sms_cp_async",
            errors,
        )

    for mlp_key, peak_key in (
        ("dram_bw_by_bytes_in_flight", "dram_bw_read_gbps"),
        ("l2_bw_by_bytes_in_flight", "l2_bw_read_gbps"),
    ):
        if mlp_key in mem:
            peak_mv = mem[peak_key]
            max_allowed = peak_mv["value"] * (1.0 + peak_mv["rel_uncertainty"])
            by_mode: dict[str, int] = {}
            for idx, pt in enumerate(mem[mlp_key]):
                mode = pt["issue_mode"]
                bif = pt["bytes_in_flight"]
                if bif != pt["active_ctas"] * pt["bytes_per_cta"]:
                    errors.append(
                        f"measured.memory.{mlp_key}[{idx}].bytes_in_flight ({bif}) != active_ctas * bytes_per_cta"
                    )
                prev = by_mode.get(mode, 0)
                if bif <= prev:
                    errors.append(
                        f"measured.memory.{mlp_key}[{idx}].bytes_in_flight ({bif}) must be strictly increasing "
                        f"within issue_mode={mode!r} (prev={prev})"
                    )
                by_mode[mode] = bif
                bw = pt["bw_gbps"]["value"]
                if bw > max_allowed:
                    errors.append(
                        f"measured.memory.{mlp_key}[{idx}].bw_gbps ({bw:.1f}) "
                        f"exceeds {peak_key} * (1 + rel_uncertainty) ({max_allowed:.1f})"
                    )

    for cl_key, peak_key in (
        ("dram_closed_loop", "dram_bw_read_gbps"),
        ("l2_closed_loop", "l2_bw_read_gbps"),
    ):
        if cl_key in mem:
            peak_mv = mem[peak_key]
            max_allowed = peak_mv["value"] * (1.0 + peak_mv["rel_uncertainty"])
            for idx, pt in enumerate(mem[cl_key]):
                tokens = pt["tokens"]
                mode = pt["issue_mode"]
                if tokens == 0 and mode != "ld_global_v4":
                    errors.append(
                        f"measured.memory.{cl_key}[{idx}]: tokens=0 requires issue_mode='ld_global_v4'"
                    )
                if tokens > 0 and mode != "cp_async":
                    errors.append(
                        f"measured.memory.{cl_key}[{idx}]: tokens={tokens} requires issue_mode='cp_async'"
                    )
                bw = pt["delivered_bw_gbps"]["value"]
                if bw > max_allowed:
                    errors.append(
                        f"measured.memory.{cl_key}[{idx}].delivered_bw_gbps ({bw:.1f}) "
                        f"exceeds {peak_key} * (1 + rel_uncertainty) ({max_allowed:.1f})"
                    )

    if "dram_closed_loop_mma" in mem:
        peak_mv = mem["dram_bw_read_gbps"]
        max_allowed = peak_mv["value"] * (1.0 + peak_mv["rel_uncertainty"])
        for idx, pt in enumerate(mem["dram_closed_loop_mma"]):
            if pt["active_ctas"] != pt["ctas_per_sm"] * sm_count:
                errors.append(
                    f"measured.memory.dram_closed_loop_mma[{idx}].active_ctas ({pt['active_ctas']}) "
                    f"!= ctas_per_sm * sm_count ({pt['ctas_per_sm'] * sm_count})"
                )
            bw = pt["delivered_bw_gbps"]["value"]
            if bw > max_allowed:
                errors.append(
                    f"measured.memory.dram_closed_loop_mma[{idx}].delivered_bw_gbps ({bw:.1f}) "
                    f"exceeds dram_bw_read_gbps * (1 + rel_uncertainty) ({max_allowed:.1f})"
                )

    if "l2_merge_by_sharing" in mem:
        dram_peak = mem["dram_bw_read_gbps"]
        l2_peak = mem["l2_bw_read_gbps"]
        max_dram = dram_peak["value"] * (1.0 + dram_peak["rel_uncertainty"])
        max_l2 = l2_peak["value"] * (1.0 + l2_peak["rel_uncertainty"])
        for idx, pt in enumerate(mem["l2_merge_by_sharing"]):
            if pt["active_ctas"] % pt["sharing"] != 0:
                errors.append(
                    f"measured.memory.l2_merge_by_sharing[{idx}]: active_ctas ({pt['active_ctas']}) "
                    f"must be divisible by sharing ({pt['sharing']})"
                )
            if pt["dram_unique_bw_gbps"]["value"] > max_dram:
                errors.append(
                    f"measured.memory.l2_merge_by_sharing[{idx}].dram_unique_bw_gbps "
                    f"({pt['dram_unique_bw_gbps']['value']:.1f}) exceeds {max_dram:.1f}"
                )
            if pt["l2_to_sm_bw_gbps"]["value"] > max_l2:
                errors.append(
                    f"measured.memory.l2_merge_by_sharing[{idx}].l2_to_sm_bw_gbps "
                    f"({pt['l2_to_sm_bw_gbps']['value']:.1f}) exceeds {max_l2:.1f}"
                )

    for lat_key, peak_key in (
        ("dram_latency_by_bw", "dram_bw_read_gbps"),
        ("l2_latency_by_bw", "l2_bw_read_gbps"),
    ):
        if lat_key in mem:
            prev_bw = -1.0
            for idx, pt in enumerate(mem[lat_key]):
                bw = pt["achieved_bw_gbps"]
                if bw < prev_bw:
                    errors.append(
                        f"measured.memory.{lat_key}[{idx}].achieved_bw_gbps ({bw}) must be non-decreasing (prev={prev_bw})"
                    )
                prev_bw = bw
            peak_mv = mem[peak_key]
            peak_bw = peak_mv["value"]
            max_allowed = peak_bw * (1.0 + peak_mv["rel_uncertainty"])
            if prev_bw > max_allowed:
                errors.append(
                    f"measured.memory.{lat_key} max achieved_bw_gbps ({prev_bw:.1f}) "
                    f"exceeds {peak_key} * (1 + rel_uncertainty) ({max_allowed:.1f})"
                )

    if "l2_rescan_hit_rate_by_working_set" in mem:
        prev_ws = 0
        for idx, pt in enumerate(mem["l2_rescan_hit_rate_by_working_set"]):
            ws = pt["working_set_bytes"]
            if ws <= prev_ws:
                errors.append(
                    f"measured.memory.l2_rescan_hit_rate_by_working_set[{idx}].working_set_bytes ({ws}) "
                    f"must be strictly increasing (prev={prev_ws})"
                )
            hr = pt["estimated_hit_rate"]["value"]
            if not (0.0 <= hr <= 1.0):
                errors.append(
                    f"measured.memory.l2_rescan_hit_rate_by_working_set[{idx}].estimated_hit_rate.value ({hr}) "
                    f"must be within [0, 1]"
                )
            prev_ws = ws

    pipeline = measured.get("pipeline")
    if isinstance(pipeline, dict) and "stage_sync_overhead_ns" in pipeline:
        for idx, pt in enumerate(pipeline["stage_sync_overhead_ns"]):
            tokens = pt["tokens"]
            mode = pt["issue_mode"]
            if tokens == 0 and mode != "ld_global_v4":
                errors.append(
                    f"measured.pipeline.stage_sync_overhead_ns[{idx}]: tokens=0 requires issue_mode='ld_global_v4'"
                )
            if tokens > 0 and mode != "cp_async":
                errors.append(
                    f"measured.pipeline.stage_sync_overhead_ns[{idx}]: tokens={tokens} requires issue_mode='cp_async'"
                )

    max_warps = specs["max_warps_per_sm"]
    for key, curve in measured["compute"]["mma_tflops_per_sm_by_warps"].items():
        _check_monotonic_curve(
            curve,
            "mma_warps_per_sm",
            max_warps,
            f"measured.compute.mma_tflops_per_sm_by_warps['{key}']",
            errors,
        )

    clock_policy = profile["provenance"]["clock_policy"]
    for field in ("observed_sm_clock_range_mhz", "observed_mem_clock_range_mhz"):
        rng = clock_policy.get(field)
        if rng is not None and rng[0] > rng[1]:
            errors.append(f"provenance.clock_policy.{field}: min ({rng[0]}) > max ({rng[1]})")

    dispatch = measured["cta_dispatch"]
    if dispatch["observed_policy"] not in dispatch["candidate_policies"]:
        errors.append(
            f"measured.cta_dispatch.observed_policy ({dispatch['observed_policy']!r}) "
            f"must appear in candidate_policies ({dispatch['candidate_policies']!r})"
        )

    return errors


def validate_profile(
    profile: dict[str, Any],
    schema_path: Path | str | None = None,
) -> None:
    """Validate a GPU profile dict against the JSON Schema and semantic invariants."""
    schema = load_schema(schema_path)
    validator = Draft202012Validator(schema)
    schema_errors = sorted(validator.iter_errors(profile), key=lambda e: list(e.absolute_path))
    if schema_errors:
        formatted = [
            f"{'.'.join(str(p) for p in err.absolute_path) or '<root>'}: {err.message}"
            for err in schema_errors
        ]
        raise ProfileValidationError(formatted)

    sem_errors = _semantic_checks(profile)
    if sem_errors:
        raise ProfileValidationError(sem_errors)


def load_profile(
    path: Path | str,
    schema_path: Path | str | None = None,
) -> dict[str, Any]:
    """Load and validate a GPU profile JSON file."""
    profile_path = Path(path)
    with profile_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    validate_profile(data, schema_path=schema_path)
    return data


def _main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m virtualgpu.profile <profile.json>", file=sys.stderr)
        return 2
    try:
        profile = load_profile(argv[1])
    except (OSError, json.JSONDecodeError, ProfileValidationError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    ident = profile["identity"]
    print(f"OK: {ident['sku']} ({ident['arch']}, schema {profile['schema_version']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
