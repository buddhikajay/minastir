import { useEffect, useMemo } from "react";
import * as THREE from "three";
import type { Pose } from "../api";
import type { GroundMap, WorldShapes } from "../sceneData";
import { makeBin, makeShrub, makeTree } from "@minastir/objects";
import { M as LIB } from "../objects";

/**
 * The ground one calibrated camera sees (road, footpath, lawn) as a flat textured layer with cars
 * removed, placed with the camera's current pose.
 */
export function GroundMapLayer({ map, pose, layer }: { map: GroundMap; pose: Pose; layer: number }) {
  const texture = useMemo(() => {
    const t = new THREE.CanvasTexture(map.canvas);
    t.colorSpace = THREE.SRGBColorSpace;
    t.magFilter = THREE.LinearFilter;
    t.minFilter = THREE.LinearMipmapLinearFilter;
    t.anisotropy = 8;
    return t;
  }, [map]);
  useEffect(() => () => texture.dispose(), [texture]);

  const span = map.half * 2;
  return (
    // Same rotation convention as camera yaw: rotation.y = yaw turns +Z toward +X.
    <group position={[pose.x, 0, pose.z]} rotation={[0, THREE.MathUtils.degToRad(pose.yaw), 0]}>
      {/* Canvas row 0 is z = -half: after laying the plane flat, its top edge points to -Z. */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.01 + layer * 0.004, 0]} renderOrder={1 + layer}>
        <planeGeometry args={[span, span]} />
        <meshStandardMaterial map={texture} transparent depthWrite={false} roughness={1} opacity={0.92} />
      </mesh>
    </group>
  );
}

// Shared geometry and materials: matte, light, restrained.
const BOX = new THREE.BoxGeometry(1, 1, 1).translate(0, 0.5, 0);
const CYL = new THREE.CylinderGeometry(1, 1, 1, 10).translate(0, 0.5, 0);
const srgb = (hex: string) => new THREE.Color(hex);
const M = {
  wall: new THREE.MeshStandardMaterial({ color: srgb("#e4e1dc"), roughness: 1 }),
  block: new THREE.MeshStandardMaterial({ color: srgb("#ecebe8"), roughness: 1 }),
  fence: new THREE.MeshStandardMaterial({ color: srgb("#c9c2b6"), roughness: 1 }),
  pole: new THREE.MeshStandardMaterial({ color: srgb("#9aa0a6"), roughness: 1 }),
};

/** Walls, house blocks, trees and poles in world coordinates (merged across cameras). */
export function ShapesLayer({ shapes }: { shapes: WorldShapes }) {
  return (
    <>
      {shapes.walls.map((w, i) => {
        const dx = w.b[0] - w.a[0];
        const dz = w.b[1] - w.a[1];
        const len = Math.hypot(dx, dz);
        if (len < 0.1) return null;
        const mid: [number, number] = [(w.a[0] + w.b[0]) / 2, (w.a[1] + w.b[1]) / 2];
        // rotation.y = θ maps local +X onto the wall and local +Z onto (−dz, dx)/len.
        const theta = Math.atan2(-dz, dx);
        const away = (-dz * w.back[0] + dx * w.back[1]) / len > 0 ? 1 : -1;
        const fence = w.label === "fence";
        const thick = fence ? 0.06 : 0.2;
        return (
          <group key={`w${i}`} position={[mid[0], 0, mid[1]]} rotation={[0, theta, 0]}>
            <mesh geometry={BOX} material={fence ? M.fence : M.wall} scale={[len, w.top, thick]} />
            {w.depth > 0 && (
              <mesh geometry={BOX} material={M.block} position={[0, 0, away * (thick / 2 + w.depth / 2)]} scale={[len, w.top, w.depth]} />
            )}
          </group>
        );
      })}
      {shapes.trees.map((t, i) => (
        <primitive
          key={`t${i}-${t.x.toFixed(2)}-${t.z.toFixed(2)}`}
          object={t.shrub ? makeShrub(LIB, t.radius) : makeTree(LIB, t.radius, Math.max(t.radius * 1.2, t.top - t.radius * 0.9))}
          position={[t.x, 0, t.z]}
        />
      ))}
      {(shapes.bins ?? []).map((b, i) => (
        <primitive key={`b${i}-${b.x.toFixed(2)}-${b.z.toFixed(2)}`} object={makeBin(LIB)} position={[b.x, 0, b.z]} />
      ))}
      {shapes.poles.map((p, i) => (
        <mesh key={`p${i}`} geometry={CYL} material={M.pole} position={[p.x, 0, p.z]} scale={[0.06, p.top, 0.06]} />
      ))}
    </>
  );
}
