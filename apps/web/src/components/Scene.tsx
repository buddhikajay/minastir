import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { ContactShadows, Line, OrbitControls, RoundedBox } from "@react-three/drei";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { api, bestProfile, type Camera, type LiveSnapshot, type Pose } from "../api";
import { LiveObjects } from "./LiveObjects";
import { GroundMapLayer, ShapesLayer } from "./GroundMapLayer";
import { StreetLayer } from "./StreetLayer";
import type { GroundMap, House, World } from "../sceneData";
import { HousesLayer } from "./HousesLayer";
import { StudioEnvironment } from "../objects";

// Tesla-style matte world.
const WORLD = "#e8e9eb";
const FLOOR = "#eef0f2";
const CONE = "#8f98a3";
const CONE_EDGE = "#b9bfc7";
const ACCENT = "#006fff";
const TARGET: [number, number, number] = [0, 0.6, 0];
const HOME_VIEW: [number, number, number] = [26, 34, 38];

const rad = (d: number) => (d * Math.PI) / 180;
const deg = (r: number) => (r * 180) / Math.PI;
export const normYaw = (d: number) => ((Math.round(d) % 360) + 360) % 360;

/** Haze proportional to how far out you are: the far edge always fades, at any zoom. */
const FOG_TARGET = new THREE.Vector3(...TARGET);
function FogRange() {
  useFrame(({ camera, scene }) => {
    const fog = scene.fog as THREE.Fog | null;
    if (!fog) return;
    const d = camera.position.distanceTo(FOG_TARGET);
    fog.near = d * 0.95;
    fog.far = d * 2.36;
  });
  return null;
}

/**
 * Frame the scene once the cameras and street are known: look from above the street back at the
 * cameras (the house front), like the console's default view. Runs once per mount.
 */
function AutoFrame({ poses, world }: { poses: Record<string, Pose>; world?: World | null }) {
  const { camera, controls } = useThree();
  const done = useRef(false);
  useEffect(() => {
    if (done.current || !controls) return;
    const ps = Object.values(poses);
    if (!ps.length) return;
    const c = new THREE.Vector3(ps.reduce((a, p) => a + p.x, 0) / ps.length, 0.6, ps.reduce((a, p) => a + p.z, 0) / ps.length);
    // Towards the street (or, without one, the way the cameras face on average).
    let dir: THREE.Vector3;
    if (world?.street) {
      const [ox, oz] = world.street.origin;
      dir = new THREE.Vector3(ox - c.x, 0, oz - c.z);
    } else {
      dir = new THREE.Vector3(
        ps.reduce((a, p) => a + Math.sin(rad(p.yaw)), 0),
        0,
        ps.reduce((a, p) => a + Math.cos(rad(p.yaw)), 0),
      );
    }
    if (dir.lengthSq() < 1e-6) return;
    dir.normalize();
    // Target a little in front of the house; stand over the far side of the street, turned a
    // quarter-octant so the view isn't square-on.
    const target = c.clone().addScaledVector(dir, 6);
    const side = new THREE.Vector3(-dir.z, 0, dir.x);
    const eye = target.clone().addScaledVector(dir, 30).addScaledVector(side, 14);
    eye.y = 30;
    camera.position.copy(eye);
    const oc = controls as unknown as { target: THREE.Vector3; update: () => void };
    oc.target.copy(target);
    oc.update();
    done.current = true;
  }, [poses, world, camera, controls]);
  return null;
}

/** Unplaced cameras sit on a ring looking at the centre (with measured tilt/height if calibrated). */
export function placeholderPose(i: number, n: number, cam?: Camera): Pose {
  const r = n <= 1 ? 5 : 6.5;
  const a = (i / Math.max(n, 1)) * Math.PI * 2 + Math.PI / 4;
  const x = +(Math.cos(a) * r).toFixed(2);
  const z = +(Math.sin(a) * r).toFixed(2);
  const cal = cam?.calibration;
  const height = cal ? +cal.height.toFixed(2) : 2.8;
  const pitch = cal ? +cal.pitch.toFixed(1) : Math.round(deg(Math.atan2(height - 0.6, r)));
  return { x, z, height, yaw: normYaw(deg(Math.atan2(-x, -z))), pitch };
}

function useLiveTexture(url: string, ms: number) {
  const [tex, setTex] = useState<THREE.Texture | null>(null);
  useEffect(() => {
    let alive = true;
    let prev: THREE.Texture | null = null;
    let timer: ReturnType<typeof setTimeout>;
    const loader = new THREE.TextureLoader();
    const tick = () =>
      loader.load(
        `${url}?t=${Date.now()}`,
        (t) => {
          if (!alive) return t.dispose();
          t.colorSpace = THREE.SRGBColorSpace;
          prev?.dispose();
          prev = t;
          setTex(t);
          timer = setTimeout(tick, ms);
        },
        undefined,
        () => {
          if (alive) timer = setTimeout(tick, ms * 3);
        },
      );
    tick();
    return () => {
      alive = false;
      clearTimeout(timer);
      prev?.dispose();
    };
  }, [url, ms]);
  return tex;
}

/** Live frame on the far plane of the selected camera's view cone. */
function ImagePlane({ id, w, h, depth }: { id: string; w: number; h: number; depth: number }) {
  const tex = useLiveTexture(api.snapshotUrl(id), 1500);
  if (!tex) return null;
  return (
    <mesh position={[0, 0, depth]} rotation={[0, Math.PI, 0]}>
      <planeGeometry args={[w * 2, h * 2]} />
      <meshBasicMaterial map={tex} side={THREE.DoubleSide} toneMapped={false} />
    </mesh>
  );
}

function CameraRig({ camera, pose, selected, onSelect }: { camera: Camera; pose: Pose; selected: boolean; onSelect: () => void }) {
  // The base group carries position + heading, so the pole stays vertical; the head adds tilt.
  const [hover, setHover] = useState(false);

  const p = bestProfile(camera);
  const aspect = p?.width && p?.height ? p.width / p.height : 16 / 9;
  const depth = 2.6;
  const h = Math.min(0.7, 1.6 / aspect); // keep panoramic cameras from producing enormous cones
  const w = h * aspect;
  const corners = useMemo(
    () =>
      [
        [-w, h, depth],
        [w, h, depth],
        [w, -h, depth],
        [-w, -h, depth],
      ] as [number, number, number][],
    [w, h],
  );
  const apex: [number, number, number] = [0, 0, 0.33];
  const cone = useMemo(() => {
    const a = [0, 0, 0.33];
    const tri = (b: number[], c: number[]) => [...a, ...b, ...c];
    const pos = new Float32Array([
      ...tri(corners[0], corners[1]),
      ...tri(corners[1], corners[2]),
      ...tri(corners[2], corners[3]),
      ...tri(corners[3], corners[0]),
    ]);
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.computeVertexNormals();
    return g;
  }, [corners]);

  const active = selected || hover;
  const tint = active ? ACCENT : CONE;
  const edge = active ? ACCENT : CONE_EDGE;

  return (
    <>
      <group position={[pose.x, pose.height, pose.z]} rotation={[0, rad(pose.yaw), 0]}>
        <mesh position={[0, -pose.height / 2, 0]}>
          <cylinderGeometry args={[0.03, 0.03, pose.height, 12]} />
          <meshStandardMaterial color="#c9cdd2" roughness={0.7} />
        </mesh>
        <mesh position={[0, -pose.height + 0.01, 0]}>
          <cylinderGeometry args={[0.16, 0.18, 0.02, 32]} />
          <meshStandardMaterial color="#d6d9dd" roughness={0.9} />
        </mesh>

        <group
          rotation={[rad(pose.pitch), 0, 0]}
          onClick={(e) => {
            e.stopPropagation();
            onSelect();
          }}
          onPointerOver={(e) => {
            e.stopPropagation();
            setHover(true);
            document.body.style.cursor = "pointer";
          }}
          onPointerOut={() => {
            setHover(false);
            document.body.style.cursor = "";
          }}
        >
          <RoundedBox args={[0.44, 0.34, 0.66]} radius={0.09} smoothness={4}>
            <meshStandardMaterial color="#f7f7f8" roughness={0.45} />
          </RoundedBox>
          <mesh position={[0, 0, 0.335]} rotation={[Math.PI / 2, 0, 0]}>
            <cylinderGeometry args={[0.11, 0.11, 0.02, 32]} />
            <meshStandardMaterial color="#2a2d31" roughness={0.25} metalness={0.4} />
          </mesh>
          <mesh geometry={cone} renderOrder={1}>
            <meshStandardMaterial
              color={tint}
              transparent
              opacity={active ? 0.16 : 0.09}
              side={THREE.DoubleSide}
              depthWrite={false}
              roughness={1}
            />
          </mesh>
          <Line points={[...corners, corners[0]]} color={edge} lineWidth={1} transparent opacity={active ? 0.9 : 0.7} />
          {corners.map((c, i) => (
            <Line key={i} points={[apex, c]} color={edge} lineWidth={1} transparent opacity={active ? 0.6 : 0.35} />
          ))}
          {selected && <ImagePlane id={camera.id} w={w} h={h} depth={depth} />}
        </group>

      </group>

    </>
  );
}

/**
 * Positions the DOM name labels over each camera every frame. (Plain DOM overlay in the app's own
 * React tree — drei's <Html> portals intermittently rendered empty.)
 */
function LabelProjector({ poses, labels }: { poses: Record<string, Pose>; labels: React.RefObject<Record<string, HTMLDivElement | null>> }) {
  const v = useRef(new THREE.Vector3()).current;
  useFrame(({ camera, size }) => {
    for (const [id, p] of Object.entries(poses)) {
      const el = labels.current?.[id];
      if (!el) continue;
      v.set(p.x, p.height + 0.5, p.z).project(camera);
      const x = (v.x * 0.5 + 0.5) * size.width;
      const y = (-v.y * 0.5 + 0.5) * size.height;
      el.style.transform = `translate(${x}px, ${y}px) translate(-50%, -100%)`;
      el.style.visibility = v.z < 1 ? "visible" : "hidden"; // behind the camera
    }
  });
  return null;
}

export function Scene({
  cameras,
  poses,
  selected,
  onSelect,
  live,
  scenes,
  world,
  houses,
}: {
  cameras: Camera[];
  poses: Record<string, Pose>;
  selected: string | null;
  onSelect: (id: string | null) => void;
  /** Server-tracked objects to draw (people, vehicles…). */
  live?: React.RefObject<LiveSnapshot | null>;
  /** What each calibrated camera sees (road, houses, trees…) as labelled points. */
  scenes?: Record<string, GroundMap>;
  world?: World | null;
  houses?: House[];
}) {
  const labels = useRef<Record<string, HTMLDivElement | null>>({});
  return (
    <div className="absolute inset-0">
    <Canvas
      camera={{ position: HOME_VIEW, fov: 36 }}
      dpr={[1, 2]}
      gl={{ antialias: true }}
      onPointerMissed={(e) => e.type === "click" && onSelect(null)}
    >
      <color attach="background" args={[WORLD]} />
      <fog attach="fog" args={[WORLD, 22, 55]} />
      <FogRange />
      <hemisphereLight args={["#ffffff", "#d5d8dc", 1.15]} />
      <directionalLight position={[6, 14, 8]} intensity={1.1} />
      <StudioEnvironment />
      <AutoFrame poses={poses} world={world} />

      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[2000, 2000]} />
        <meshStandardMaterial color={FLOOR} roughness={1} />
      </mesh>
      <ContactShadows position={[0, 0.005, 0]} opacity={0.32} scale={34} blur={2.8} far={3.4} resolution={1024} />

      {cameras.map((c) =>
        poses[c.id] ? (
          <CameraRig
            key={c.id}
            camera={c}
            pose={poses[c.id]}
            selected={selected === c.id}
            onSelect={() => onSelect(c.id)}
          />
        ) : null,
      )}

      {scenes &&
        cameras.map((c, i) =>
          scenes[c.id] && poses[c.id] ? <GroundMapLayer key={`gm-${c.id}`} map={scenes[c.id]} pose={poses[c.id]} layer={i} /> : null,
        )}
      {world?.street && <StreetLayer street={world.street} />}
      {world && <ShapesLayer shapes={world.shapes} />}
      {houses && <HousesLayer houses={houses} />}
      {live && <LiveObjects live={live} />}

      <OrbitControls
        makeDefault
        enableDamping
        dampingFactor={0.08}
        maxPolarAngle={Math.PI / 2.15}
        minDistance={4}
        maxDistance={400}
        target={TARGET}
      />
      <LabelProjector poses={poses} labels={labels} />
    </Canvas>
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        {cameras.map((c) => (
          <div
            key={c.id}
            ref={(el) => {
              labels.current[c.id] = el;
            }}
            className="absolute left-0 top-0 will-change-transform"
            style={{ visibility: "hidden" }}
          >
            <div
              className={`flex select-none items-center gap-1.5 whitespace-nowrap rounded-md bg-surface px-2 py-1 text-[11px] font-medium shadow-[var(--shadow-float)] ${
                selected === c.id ? "text-accent" : "text-fg"
              }`}
            >
              <span className="h-1.5 w-1.5 rounded-full bg-ok" />
              {c.label}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
