"""CPU-only tests of the sync-load-reuse driver and its long-launch registration."""

from microbench import long_launch
from microbench import sync_load_reuse as slr

POINTS = [
    (level, ctas, sharing, load)
    for level in ("l2", "dram")
    for ctas in (1, 2)
    for sharing in ("disjoint", "shared")
    for load in ("default", "cg")
    if not (ctas == 1 and sharing == "shared")
]


def head(k):
    return {"level": k[0], "ctas_per_sm": k[1], "sharing": k[2], "load": k[3]}


def test_freeze_keys_points_by_level_ctas_sharing_and_load():
    pilot = {
        "samples": 3,
        "points": [head(k) | {"steps": 500, "launch_ns_samples": [1.3e6] * 3} for k in POINTS],
    }
    schedule, record = long_launch.freeze("sync-load-reuse", pilot)
    assert len(schedule.split()) == 12
    assert record["points"][0]["point"] == {"field": "points", **head(POINTS[0])}


def capture(step_ns, observed=None):
    runs = []
    for _ in range(3):
        pts = []
        for k in POINTS:
            t = step_ns(k)
            pts.append(
                head(k)
                | {
                    "steps": 1000,
                    "launch_ns_samples": [t * 1000] * 15,
                    "ns_per_step_samples": [t] * 15,
                    "ctas_per_sm_observed": observed(k) if observed else [k[1], k[1]],
                }
            )
        runs.append({"valid": True, "raw": {"samples": 15, "points": pts}})
    return {"runs": runs}


def reuse(k):
    level, ctas, sharing, load = k
    base = 1000.0 if level == "l2" else 3000.0
    if ctas == 2 and sharing == "shared" and load == "default":
        return base / 1.10
    return base


def test_reuse_only_with_default_loads_is_shown():
    out = slr.judge(capture(reuse))
    assert out["candidate_1_l1_reuse"] == "shown"
    assert out["levels"]["l2"]["default_path"] == "does not differ"


def test_no_difference_is_not_shown():
    out = slr.judge(capture(lambda k: 1000.0))
    assert out["candidate_1_l1_reuse"] == "not shown"


def test_cg_also_faster_shared_is_not_resolved():
    def both(k):
        return 1000.0 / 1.10 if (k[1] == 2 and k[2] == "shared") else 1000.0

    assert slr.judge(capture(both))["candidate_1_l1_reuse"] == "not resolved"


def test_ctas_not_two_per_sm_leave_the_level_unresolved():
    def placed(k):
        return [1, 3] if k == ("l2", 2, "shared", "cg") else [k[1], k[1]]

    out = slr.judge(capture(reuse, observed=placed))
    assert out["candidate_1_l1_reuse"].startswith("not resolved")
    assert out["levels"]["l2"]["failing_points"] == {"l2/2/shared/cg": "co-residency"}


def test_default_path_differs_at_one_cta():
    def slower(k):
        return 1100.0 if (k[1] == 1 and k[3] == "default") else 1000.0

    assert slr.judge(capture(slower))["levels"]["l2"]["default_path"] == "differs"


def test_fewer_than_three_valid_processes_is_not_decided():
    rec = capture(reuse)
    rec["runs"][0]["valid"] = False
    assert slr.judge(rec)["status"] == "not decided"
