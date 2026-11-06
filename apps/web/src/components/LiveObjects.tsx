import { useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { blob, makeBin, makeCar, makePerson, makePool, softBox } from "@minastir/objects";
import type { LiveObject, LiveSnapshot } from "../api";
import { M, MODE, ownMaterials } from "../objects";

/**
 * What the server is tracking, drawn with the Minastir object library: sculpted cars, smooth
 * people in a soft pool, that glide between the ~10 Hz updates, turn to face where they're going,
 * and fade in and out. All detection/tracking happens on the server; this only draws its output.
 */

const FADE = 0.5; // seconds to fade out after the server stops reporting an object

type Seen = { obj: LiveObject; lastSeen: number };

export function LiveObjects({ live }: { live: React.RefObject<LiveSnapshot | null> }) {
  const seen = useRef(new Map<number, Seen>());
  const lastTime = useRef(0);
  const [ids, setIds] = useState<number[]>([]);

  useFrame(({ clock }) => {
    const now = clock.elapsedTime;
    const snap = live.current;
    if (snap && snap.time !== lastTime.current) {
      lastTime.current = snap.time;
      for (const o of snap.objects) seen.current.set(o.id, { obj: o, lastSeen: now });
    }
    for (const [id, s] of seen.current) if (now - s.lastSeen > FADE) seen.current.delete(id);
    const next = [...seen.current.keys()].sort((a, b) => a - b);
    if (next.length !== ids.length || next.some((id, i) => id !== ids[i])) setIds(next);
  });

  return (
    <>
      {ids.map((id) => (
        <Actor key={id} id={id} seen={seen} />
      ))}
    </>
  );
}

function Actor({ id, seen }: { id: number; seen: React.RefObject<Map<number, Seen>> }) {
  const group = useRef<THREE.Group>(null);
  const first = useRef(true);
  const opacity = useRef(0);
  const initial = seen.current.get(id)?.obj;
  const kind = initial?.class ?? "person";
  // Each actor owns its materials so it can fade on its own.
  const { object, materials } = useMemo(() => {
    const object = build(kind);
    return { object, materials: ownMaterials(object) };
  }, [kind]);
  useEffect(() => () => materials.forEach((m) => m.dispose()), [materials]);

  useFrame(({ clock }, dt) => {
    const g = group.current;
    const s = seen.current.get(id);
    if (!g || !s) return;
    const { obj } = s;
    const target = new THREE.Vector3(obj.x, 0, obj.z);
    if (first.current) {
      g.position.copy(target);
      g.rotation.y = THREE.MathUtils.degToRad(obj.heading);
      first.current = false;
    }
    // Ease toward the newest position (critically-damped feel at any frame rate) — except a
    // parked object corrected to a new spot (e.g. its fitted pose arriving), which jumps there
    // instead of appearing to drive.
    if (obj.speed < 0.35 && g.position.distanceTo(target) > 0.8) g.position.copy(target);
    const k = 1 - Math.exp(-dt * 8);
    g.position.lerp(target, k);
    // Face the server's heading whenever it's known (from moving, or fitted while parked); a
    // parked object that doesn't know which way it faces keeps its orientation.
    if (obj.speed > 0.35 || obj.heading_known) {
      const want = THREE.MathUtils.degToRad(obj.heading);
      let d = want - g.rotation.y;
      d = Math.atan2(Math.sin(d), Math.cos(d));
      g.rotation.y += d * (1 - Math.exp(-dt * 6));
    }
    const gone = clock.elapsedTime - s.lastSeen > 0.25;
    opacity.current += ((gone ? 0 : 1) - opacity.current) * (1 - Math.exp(-dt * 10));
    for (const m of materials) m.opacity = (m.userData.baseOpacity as number) * opacity.current;
  });

  return (
    <group ref={group}>
      <primitive object={object} />
    </group>
  );
}

/**
 * Library shapes in the actor's frame, where local +Z is the direction of travel. Library cars
 * face +X, so they're turned a quarter.
 */
function build(kind: LiveObject["class"]): THREE.Object3D {
  const root = new THREE.Group();
  switch (kind) {
    case "car":
    case "truck":
    case "bus": {
      const car = makeCar(M, kind === "car" ? M.paintA : M.paintB);
      car.rotation.y = -Math.PI / 2;
      // A truck or bus: the same body, longer and taller.
      if (kind !== "car") car.scale.set(kind === "bus" ? 2.2 : 1.35, kind === "bus" ? 1.9 : 1.4, 1.2);
      root.add(car);
      break;
    }
    case "bin":
      // A "vehicle" the describer recognised as a wheelie bin.
      root.add(makeBin(M));
      break;
    case "bicycle":
    case "motorcycle": {
      // No rider: a person on it is detected and drawn separately.
      root.add(softBox(0.35, 0.75, kind === "bicycle" ? 1.7 : 2.0, 0.12, M.paintB));
      blob(root, 0.9, 2.2, 0.6);
      break;
    }
    case "cat":
    case "dog": {
      const body = new THREE.Mesh(new THREE.CapsuleGeometry(0.15, 0.45, 4, 12), M.person);
      body.rotation.x = Math.PI / 2;
      body.position.y = 0.3;
      root.add(body);
      blob(root, 0.8, 0.8, 0.5);
      break;
    }
    default: {
      const p = makePerson(M);
      p.add(makePool(MODE, 3));
      root.add(p);
    }
  }
  return root;
}
