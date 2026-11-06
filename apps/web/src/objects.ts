import { useThree } from "@react-three/fiber";
import { useEffect } from "react";
import * as THREE from "three";
import { applyMode, createMaterials, studioEnvironment, type Mode } from "@minastir/objects";

/**
 * The app's one set of object materials (cars, people, trees, houses) from the shared Minastir
 * object library. The app's chrome is light for now, so the 3D objects use the Day palette.
 */
export const MODE: Mode = "day";
export const M = createMaterials();
applyMode(M, MODE);

/** Soft studio reflections for paint and glass (sets the scene's environment map). */
export function StudioEnvironment() {
  const { gl, scene } = useThree();
  useEffect(() => {
    const env = studioEnvironment(gl, MODE);
    scene.environment = env;
    return () => {
      scene.environment = null;
      env.dispose();
    };
  }, [gl, scene]);
  return null;
}

/**
 * Give an object its own copies of its materials, all transparent, so it can fade in and out on
 * its own. Returns the copies.
 */
export function ownMaterials(root: THREE.Object3D): THREE.Material[] {
  const out: THREE.Material[] = [];
  root.traverse((o) => {
    const m = o as THREE.Mesh;
    if (!m.isMesh) return;
    const copy = (mat: THREE.Material) => {
      const c = mat.clone();
      c.transparent = true;
      c.userData.baseOpacity = mat.opacity;
      out.push(c);
      return c;
    };
    m.material = Array.isArray(m.material) ? m.material.map(copy) : copy(m.material);
  });
  return out;
}
