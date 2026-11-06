#!/usr/bin/env python3
"""Score a logged replay run against what is known to be true in the footage.

  python3 tools/record/check_replay.py <replay dir with truth.json> <snapshots.jsonl> [pass seconds]

truth.json lists objects known to stay parked the whole time ("parked": name, class, near [x, z],
radius) and roughly how many people there are ("people": {"expected": n}). The log is one live
snapshot per line (as written by a WebSocket logger). For each parked object it reports how many
track ids it got, how far its shown position wandered, how much its heading changed, its top speed
and whether it ever looked like it was moving; for people, how many ids appeared versus expected.
Exit status 1 if a parked object failed.
"""
import json
import math
import sys


def main():
    folder, log = sys.argv[1], sys.argv[2]
    span = float(sys.argv[3]) if len(sys.argv) > 3 else None
    truth = json.load(open(f"{folder}/truth.json"))
    snaps = [json.loads(l) for l in open(log)]
    t0 = snaps[0]["t"]
    if span:
        snaps = [s for s in snaps if s["t"] - t0 < span]
    ok = True
    for p in truth.get("parked", []):
        rows = [
            (s["t"] - t0, o)
            for s in snaps
            for o in s["objects"]
            if o["class"] in ("car", "truck", "bus") and math.hypot(o["x"] - p["near"][0], o["z"] - p["near"][1]) < p["radius"]
        ]
        if not rows:
            print(f"✗ {p['name']}: never seen")
            ok = False
            continue
        # Judge once it settled: from its first fitted pose (or after 5 s if it never got one).
        first_fit = next((t for t, o in rows if o.get("fit") is not None), 5.0)
        settled = [o for t, o in rows if t >= first_fit] or [o for _, o in rows]
        ids = sorted({o["id"] for _, o in rows})
        xs, zs = [o["x"] for o in settled], [o["z"] for o in settled]
        wander = math.hypot(max(xs) - min(xs), max(zs) - min(zs))
        hs = [o["heading"] for o in settled]
        turn = max(abs((a - hs[0] + 180) % 360 - 180) for a in hs)
        top = max(o["speed"] for _, o in rows)
        moving = sum(1 for _, o in rows if o["speed"] > 0.35)
        fits = [o.get("fit") for o in settled if o.get("fit") is not None]
        good = len(ids) == 1 and wander < 0.5 and turn < 15 and moving == 0
        ok &= good
        print(f"{'✓' if good else '✗'} {p['name']}: ids {ids}, wandered {wander:.2f} m, heading changed {turn:.0f}°, "
              f"top speed {top:.2f} m/s, moving in {moving}/{len(rows)} snapshots, fit {min(fits):.2f}–{max(fits):.2f}" if fits else
              f"{'✓' if good else '✗'} {p['name']}: ids {ids}, wandered {wander:.2f} m, heading changed {turn:.0f}°, top speed {top:.2f} m/s, never fitted")
    people = truth.get("people")
    if people:
        ids = {o["id"] for s in snaps for o in s["objects"] if o["class"] == "person"}
        shown = sum(sum(1 for o in s["objects"] if o["class"] == "person") for s in snaps) / max(len(snaps), 1)
        print(f"• people: {len(ids)} ids for about {people['expected']} people (on average {shown:.1f} shown at once)")
        cams = {}
        for snap in snaps:
            for o in snap["objects"]:
                if o["class"] == "person":
                    cams.setdefault(o["id"], set()).update(o["cameras"])
        by = {}
        for cs in cams.values():
            key = "+".join(sorted(c[:4] for c in cs)) or "(none)"
            by[key] = by.get(key, 0) + 1
        print("  ids by cameras that saw them: " + ", ".join(f"{k} {v}" for k, v in sorted(by.items())))
    dogs = truth.get("dogs")
    if dogs:
        ids = {o["id"] for s in snaps for o in s["objects"] if o["class"] in ("dog", "cat")}
        print(f"• animals: {len(ids)} ids for {dogs['expected']} dog(s)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
