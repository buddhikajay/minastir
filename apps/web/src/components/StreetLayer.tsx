import * as THREE from "three";
import { DRIVEWAY_RGB, LABELS, LABEL_RGB, type Street } from "../sceneData";

const KERB = "#f3f3f1";
const driveway = new THREE.Color().setRGB(DRIVEWAY_RGB[0] / 255, DRIVEWAY_RGB[1] / 255, DRIVEWAY_RGB[2] / 255, THREE.SRGBColorSpace);

/**
 * The fitted street: road, nature strips and footpaths as crisp flat strips, with low kerbs along
 * the road's edges. Drawn just above the per-camera ground maps, so it covers their noise.
 */
export function StreetLayer({ street }: { street: Street }) {
  const rgb = (label: string) => {
    const c = LABEL_RGB[LABELS.indexOf(label as (typeof LABELS)[number])] ?? LABEL_RGB[0];
    return new THREE.Color().setRGB(c[0] / 255, c[1] / 255, c[2] / 255, THREE.SRGBColorSpace);
  };
  return (
    // Local frame: x = across (t), z = along (s); rotation.y = heading turns +Z along the street.
    <group position={[street.origin[0], 0, street.origin[1]]} rotation={[0, THREE.MathUtils.degToRad(street.heading), 0]}>
      {street.bands.map((b, i) => (
        <group key={i}>
          <mesh rotation={[-Math.PI / 2, 0, 0]} position={[(b.t0 + b.t1) / 2, 0.03, (b.s0 + b.s1) / 2]} renderOrder={10}>
            <planeGeometry args={[b.t1 - b.t0, b.s1 - b.s0]} />
            <meshStandardMaterial color={rgb(b.label)} roughness={1} />
          </mesh>
          {b.label === "road" &&
            [b.t0, b.t1].map((t, k) => (
              <mesh key={k} position={[t + (k === 0 ? -0.08 : 0.08), 0.08, (b.s0 + b.s1) / 2]}>
                <boxGeometry args={[0.16, 0.1, b.s1 - b.s0]} />
                <meshStandardMaterial color={KERB} roughness={1} />
              </mesh>
            ))}
        </group>
      ))}
      {street.driveways.map((d, i) => (
        <mesh key={`d${i}`} rotation={[-Math.PI / 2, 0, 0]} position={[(d.t0 + d.t1) / 2, 0.04, (d.s0 + d.s1) / 2]} renderOrder={11}>
          <planeGeometry args={[d.t1 - d.t0, d.s1 - d.s0]} />
          <meshStandardMaterial color={driveway} roughness={1} />
        </mesh>
      ))}
    </group>
  );
}
