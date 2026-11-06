import { useEffect, useMemo } from "react";
import type * as THREE from "three";
import { makeHouseFromOutline } from "@minastir/objects";
import type { House } from "../sceneData";
import { M } from "../objects";

/** Houses traced on the site plan: soft walls and a hipped roof (object library). */
export function HousesLayer({ houses }: { houses: House[] }) {
  const groups = useMemo(() => houses.map((h) => makeHouseFromOutline(M, h.points, Math.max(2.4, h.height - 1.6), 1.6)), [houses]);
  useEffect(
    () => () =>
      groups.forEach((g) =>
        g.traverse((o) => {
          const m = o as THREE.Mesh;
          if (m.isMesh) m.geometry.dispose();
        }),
      ),
    [groups],
  );
  return (
    <>
      {groups.map((g, i) => (
        <primitive key={i} object={g} />
      ))}
    </>
  );
}
