/**
 * Minastir object library: the things the 3D map shows — cars, people, trees, houses and
 * cameras — in the brand's quiet, sculpted style (brand/reference/Minastir-Console.html):
 * smooth matte surfaces, clearcoat paint, soft contact shadows, studio reflections, Night and
 * Day palettes. Builders return plain three.js objects in a local frame:
 *   x forward (cars) / along the ridge (houses), y up, z across; origin on the ground.
 * Shared by the website hero and the app's 3D view.
 */
import * as THREE from "three";

export type Mode = "night" | "day";

// ---------------------------------------------------------------- palettes (brand § 03)

export const PALETTE = {
  night: {
    bg: 0x0e0f11,
    fog: [0x0e0f11, 50, 120] as [number, number, number],
    envI: 0.9,
    ground: ["#2a2d32", "#1a1c20", "#0e0f11"],
    road: 0x1f2125,
    lane: 0x8a8f96,
    laneOp: 0.55,
    wall: 0x9a9fa6,
    roof: 0x4b4f56,
    foliage: 0x3d4147,
    trim: 0x55595f,
    window: 0x0b0c0e,
    paintA: 0xa9aeb5,
    paintB: 0xe6e8eb,
    glass: 0x0c0d0f,
    tyre: 0x111214,
    rim: 0x6b7078,
    person: 0xffffff,
    personE: 0x8a8a8a,
    pole: 0x8a8f96,
    hemi: [0xc8ced8, 0x0e0f11, 0.45] as [number, number, number],
    key: 0.55,
    fan: 0xffffff,
    fanOp: 0.07,
    sel: 0x4c9bff,
    selOp: 0.3,
    line: 0xffffff,
    lineOp: 0.32,
    blind: 0xf2665e,
    blindOp: 0.18,
    pool: 0.75,
    poolAdditive: true,
    poolCol: 0xffffff,
    pLight: 0.9,
    blobOp: 1,
  },
  day: {
    bg: 0xcdcfd4,
    fog: [0xcdcfd4, 60, 140] as [number, number, number],
    envI: 0.8,
    ground: ["#e2e4e8", "#cfd2d7", "#bfc2c7"],
    road: 0xb4b7bd,
    lane: 0xffffff,
    laneOp: 0.9,
    wall: 0xf1f2f4,
    roof: 0x8f949b,
    foliage: 0xc9ccd1,
    trim: 0x9a9fa6,
    window: 0x3a3e44,
    paintA: 0x5e636b,
    paintB: 0x2a2d32,
    glass: 0x111316,
    tyre: 0x1b1d21,
    rim: 0x9a9fa6,
    person: 0x3a3e44,
    personE: 0x000000,
    pole: 0x5e636b,
    hemi: [0xffffff, 0x8a8f96, 0.55] as [number, number, number],
    key: 0.7,
    fan: 0x111316,
    fanOp: 0.06,
    sel: 0x1f63d6,
    selOp: 0.5,
    line: 0x111316,
    lineOp: 0.28,
    blind: 0xc62f27,
    blindOp: 0.18,
    pool: 0.22,
    poolAdditive: false,
    poolCol: 0x111316,
    pLight: 0,
    blobOp: 0.55,
  },
};

// ---------------------------------------------------------------- textures

/** Canvas radial gradient texture (browser only). */
export function radialTexture(stops: [number, string][], size = 512): THREE.CanvasTexture {
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const g = c.getContext("2d")!;
  const r = g.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  stops.forEach(([o, s]) => r.addColorStop(o, s));
  g.fillStyle = r;
  g.fillRect(0, 0, size, size);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

let blobTex: THREE.Texture | null = null;
let poolTex: THREE.Texture | null = null;
const blobMaterials: THREE.MeshBasicMaterial[] = [];

/** Soft dark ellipse on the ground under an object (contact shadow). */
export function blob(parent: THREE.Object3D, w: number, d: number, opacity = 0.75): THREE.Mesh {
  blobTex ??= radialTexture(
    [
      [0, "rgba(0,0,0,0.85)"],
      [0.45, "rgba(0,0,0,0.45)"],
      [1, "rgba(0,0,0,0)"],
    ],
    256,
  );
  const mat = new THREE.MeshBasicMaterial({ map: blobTex, transparent: true, depthWrite: false, opacity });
  mat.userData.baseOpacity = opacity;
  blobMaterials.push(mat);
  const m = new THREE.Mesh(new THREE.PlaneGeometry(w, d), mat);
  m.rotation.x = -Math.PI / 2;
  m.position.y = 0.012;
  m.renderOrder = 1;
  parent.add(m);
  return m;
}

// ---------------------------------------------------------------- materials

export type Materials = ReturnType<typeof createMaterials>;

/** One shared set of materials; `applyMode` recolours them for Night / Day. */
export function createMaterials() {
  return {
    ground: new THREE.MeshStandardMaterial({ roughness: 1 }),
    road: new THREE.MeshStandardMaterial({ roughness: 0.95 }),
    wall: new THREE.MeshStandardMaterial({ roughness: 0.8 }),
    roof: new THREE.MeshStandardMaterial({ roughness: 0.6 }),
    foliage: new THREE.MeshStandardMaterial({ roughness: 0.9 }),
    trim: new THREE.MeshStandardMaterial({ roughness: 0.7 }),
    window: new THREE.MeshPhysicalMaterial({ roughness: 0.12, metalness: 0.2, clearcoat: 1 }),
    paintA: new THREE.MeshPhysicalMaterial({ metalness: 0.35, roughness: 0.38, clearcoat: 1, clearcoatRoughness: 0.08 }),
    paintB: new THREE.MeshPhysicalMaterial({ metalness: 0.35, roughness: 0.38, clearcoat: 1, clearcoatRoughness: 0.08 }),
    glass: new THREE.MeshPhysicalMaterial({ roughness: 0.08, metalness: 0.3, clearcoat: 1, clearcoatRoughness: 0.03 }),
    tyre: new THREE.MeshStandardMaterial({ roughness: 0.85 }),
    rim: new THREE.MeshStandardMaterial({ metalness: 0.8, roughness: 0.3 }),
    tail: new THREE.MeshStandardMaterial({ color: 0x7a1410, emissive: 0xd8261c, emissiveIntensity: 1.2 }),
    head: new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xffffff, emissiveIntensity: 0.9 }),
    person: new THREE.MeshPhysicalMaterial({ roughness: 0.35, clearcoat: 0.5 }),
    camBody: new THREE.MeshPhysicalMaterial({ color: 0xf2f3f5, roughness: 0.25, clearcoat: 1 }),
    pole: new THREE.MeshStandardMaterial({ roughness: 0.5, metalness: 0.4 }),
    lane: new THREE.MeshBasicMaterial({ transparent: true }),
  };
}

const groundTex: Partial<Record<Mode, THREE.Texture>> = {};

/** Recolour the shared materials (and contact shadows) for a mode. */
export function applyMode(M: Materials, mode: Mode) {
  const p = PALETTE[mode];
  groundTex[mode] ??= radialTexture([
    [0, p.ground[0]],
    [0.45, p.ground[1]],
    [1, p.ground[2]],
  ]);
  M.ground.map = groundTex[mode]!;
  M.ground.color.set(0xffffff);
  M.ground.needsUpdate = true;
  const keys = ["road", "wall", "roof", "foliage", "trim", "window", "paintA", "paintB", "glass", "tyre", "rim", "person", "pole"] as const;
  for (const k of keys) M[k].color.set(p[k]);
  for (const mt of Object.values(M)) if ("envMapIntensity" in mt) (mt as THREE.MeshStandardMaterial).envMapIntensity = p.envI;
  M.person.emissive.set(p.personE);
  M.lane.color.set(p.lane);
  M.lane.opacity = p.laneOp;
  for (const b of blobMaterials) b.opacity = (b.userData.baseOpacity as number) * p.blobOp;
}

/** Soft studio environment for gentle, car-like reflections. */
export function studioEnvironment(renderer: THREE.WebGLRenderer, mode: Mode): THREE.Texture {
  const day = mode === "day";
  const env = new THREE.Scene();
  env.add(new THREE.Mesh(new THREE.BoxGeometry(40, 20, 40), new THREE.MeshBasicMaterial({ color: day ? 0x9a9da3 : 0x1b1d21, side: THREE.BackSide })));
  const panel = (w: number, h: number, x: number, y: number, z: number, rx: number, ry: number, k: number) => {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ color: new THREE.Color(k, k, k), side: THREE.DoubleSide }));
    m.position.set(x, y, z);
    m.rotation.set(rx, ry, 0);
    env.add(m);
  };
  panel(24, 6, 0, 9.5, 0, Math.PI / 2, 0, day ? 3.0 : 1.6); // long overhead softbox
  panel(10, 6, -19.5, 4, 0, 0, Math.PI / 2, day ? 1.6 : 0.6); // side fill
  panel(10, 6, 19.5, 4, 4, 0, -Math.PI / 2, day ? 1.2 : 0.35); // rim
  const pm = new THREE.PMREMGenerator(renderer);
  const tex = pm.fromScene(env, 0.04).texture;
  pm.dispose();
  return tex;
}

// ---------------------------------------------------------------- shapes

/** Box with softly bevelled edges (walls, benches, bins), standing on the ground. */
export function softBox(w: number, h: number, d: number, r: number, mat: THREE.Material): THREE.Mesh {
  const s = new THREE.Shape();
  const x = -w / 2 + r,
    z = -d / 2 + r,
    iw = w - 2 * r,
    id = d - 2 * r;
  s.moveTo(x, z);
  s.lineTo(x + iw, z);
  s.lineTo(x + iw, z + id);
  s.lineTo(x, z + id);
  s.closePath();
  const bev = Math.min(r, h / 2 - 0.001);
  const g = new THREE.ExtrudeGeometry(s, {
    depth: Math.max(0.001, h - 2 * bev),
    bevelEnabled: true,
    bevelThickness: bev,
    bevelSize: r,
    bevelSegments: 5,
    curveSegments: 8,
  });
  g.rotateX(-Math.PI / 2);
  g.translate(0, bev, 0);
  g.computeVertexNormals();
  return new THREE.Mesh(g, mat);
}

function shapeVerts(geo: THREE.BufferGeometry, fn: (v: THREE.Vector3) => void) {
  const p = geo.attributes.position as THREE.BufferAttribute;
  const v = new THREE.Vector3();
  for (let i = 0; i < p.count; i++) {
    v.fromBufferAttribute(p, i);
    fn(v);
    p.setXYZ(i, v.x, v.y, v.z);
  }
  geo.computeVertexNormals();
}

/** A car, 4.9 m long, facing +x: sculpted side profile, shaped in plan and section. */
export function makeCar(M: Materials, paint: THREE.Material = M.paintA): THREE.Group {
  const car = new THREE.Group();
  const WID = 1.92,
    HL = 2.45;
  const s = new THREE.Shape();
  s.moveTo(2.3, 0.3);
  s.lineTo(1.92, 0.3);
  s.absarc(1.45, 0.3, 0.47, 0, Math.PI, false);
  s.lineTo(-0.98, 0.3);
  s.absarc(-1.45, 0.3, 0.47, 0, Math.PI, false);
  s.lineTo(-2.3, 0.3);
  s.quadraticCurveTo(-2.46, 0.42, -2.44, 0.7);
  s.quadraticCurveTo(-2.4, 0.98, -2.02, 1.02);
  s.lineTo(-1.1, 1.02);
  s.lineTo(1.0, 0.97);
  s.quadraticCurveTo(1.7, 0.93, 2.22, 0.78);
  s.quadraticCurveTo(2.5, 0.66, 2.46, 0.46);
  s.quadraticCurveTo(2.42, 0.31, 2.3, 0.3);
  const bodyG = new THREE.ExtrudeGeometry(s, { depth: WID - 0.3, bevelEnabled: true, bevelThickness: 0.15, bevelSize: 0.1, bevelSegments: 6, curveSegments: 20, steps: 1 });
  bodyG.translate(0, 0, -(WID - 0.3) / 2);
  shapeVerts(bodyG, (v) => {
    const u = Math.abs(v.x) / HL;
    const plan = 1 - 0.2 * Math.pow(Math.max(0, (u - 0.72) / 0.28), 2);
    const tumble = 1 - 0.1 * THREE.MathUtils.clamp((v.y - 0.62) / 0.42, 0, 1);
    v.z *= plan * tumble;
  });
  car.add(new THREE.Mesh(bodyG, paint));

  const g = new THREE.Shape();
  g.moveTo(1.04, 0.95);
  g.quadraticCurveTo(0.55, 1.3, 0.05, 1.5);
  g.quadraticCurveTo(-0.55, 1.62, -1.15, 1.56);
  g.quadraticCurveTo(-1.85, 1.38, -2.12, 1.0);
  g.lineTo(1.04, 0.95);
  const glassG = new THREE.ExtrudeGeometry(g, { depth: WID - 0.62, bevelEnabled: true, bevelThickness: 0.14, bevelSize: 0.12, bevelSegments: 6, curveSegments: 20 });
  glassG.translate(0, 0, -(WID - 0.62) / 2);
  shapeVerts(glassG, (v) => {
    v.z *= 1 - 0.22 * THREE.MathUtils.clamp((v.y - 0.98) / 0.6, 0, 1);
  });
  car.add(new THREE.Mesh(glassG, M.glass));

  const tyreG = new THREE.TorusGeometry(0.27, 0.11, 16, 40);
  const rimG = new THREE.CylinderGeometry(0.25, 0.25, 0.16, 40);
  rimG.rotateX(Math.PI / 2);
  for (const [x, sd] of [
    [1.45, 1],
    [1.45, -1],
    [-1.45, 1],
    [-1.45, -1],
  ]) {
    const z = sd * (WID / 2 - 0.16);
    const t = new THREE.Mesh(tyreG, M.tyre);
    t.position.set(x, 0.38, z);
    car.add(t);
    const r = new THREE.Mesh(rimG, M.rim);
    r.position.set(x, 0.38, z + sd * 0.03);
    car.add(r);
  }
  const tail = softBox(0.06, 0.07, WID * 0.78, 0.03, M.tail);
  tail.position.set(-2.43, 0.86, 0);
  car.add(tail);
  const head = softBox(0.06, 0.05, WID * 0.7, 0.025, M.head);
  head.position.set(2.42, 0.68, 0);
  car.add(head);
  blob(car, 6.2, 3.0, 0.9);
  return car;
}

/** A pedestrian, 1.8 m: smooth lathed silhouette with a head. */
export function makePerson(M: Materials): THREE.Group {
  const pr: [number, number][] = [
    [0, 0],
    [0.1, 0.01],
    [0.12, 0.05],
    [0.11, 0.45],
    [0.15, 0.82],
    [0.19, 0.98],
    [0.23, 1.25],
    [0.24, 1.38],
    [0.2, 1.46],
    [0.08, 1.5],
    [0.065, 1.56],
    [0, 1.57],
  ];
  const body = new THREE.Mesh(new THREE.LatheGeometry(pr.map(([r, h]) => new THREE.Vector2(r, h)), 40), M.person);
  body.scale.set(1, 1, 0.72);
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.115, 32, 24), M.person);
  head.position.y = 1.69;
  const grp = new THREE.Group();
  grp.add(body, head);
  blob(grp, 1.0, 1.0, 0.6);
  return grp;
}

/** A soft pool of light on the ground around a person (Night) or a gentle shadow (Day). */
export function makePool(mode: Mode, size = 6): THREE.Mesh {
  poolTex ??= radialTexture(
    [
      [0, "rgba(255,255,255,1)"],
      [0.3, "rgba(255,255,255,0.45)"],
      [1, "rgba(255,255,255,0)"],
    ],
    256,
  );
  const p = PALETTE[mode];
  const m = new THREE.Mesh(
    new THREE.PlaneGeometry(size, size),
    new THREE.MeshBasicMaterial({
      map: poolTex,
      transparent: true,
      depthWrite: false,
      opacity: p.pool,
      color: p.poolCol,
      blending: p.poolAdditive ? THREE.AdditiveBlending : THREE.NormalBlending,
    }),
  );
  m.rotation.x = -Math.PI / 2;
  m.position.y = 0.03;
  m.renderOrder = 1;
  return m;
}

const crownG = new THREE.SphereGeometry(1, 32, 24);

/** A tree: tapered trunk and a soft clustered crown of radius `r`, crown centre at height `h`. */
export function makeTree(M: Materials, r: number, h: number): THREE.Group {
  const g = new THREE.Group();
  const trunkH = Math.max(0.6, h - r * 0.5);
  const trunk = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.2, trunkH, 12), M.trim);
  trunk.position.y = trunkH / 2;
  g.add(trunk);
  for (const [dx, dy, dz, k] of [
    [0, 0, 0, 1],
    [0.45, -0.25, 0.3, 0.72],
    [-0.4, -0.3, -0.25, 0.7],
    [0.1, 0.35, -0.35, 0.65],
  ]) {
    const c = new THREE.Mesh(crownG, M.foliage);
    c.position.set(dx * r, h + dy * r, dz * r);
    c.scale.setScalar(r * k);
    g.add(c);
  }
  blob(g, r * 3.2, r * 3.2, 0.65);
  return g;
}

/** A shrub: a low clustered crown with no trunk. */
export function makeShrub(M: Materials, r: number): THREE.Group {
  const g = new THREE.Group();
  for (const [dx, dz, k] of [
    [0, 0, 1],
    [0.5, 0.25, 0.7],
    [-0.45, -0.2, 0.72],
  ]) {
    const c = new THREE.Mesh(crownG, M.foliage);
    c.position.set(dx * r, r * 0.7 * k, dz * r);
    c.scale.set(r * k, r * 0.75 * k, r * k);
    g.add(c);
  }
  blob(g, r * 2.6, r * 2.6, 0.55);
  return g;
}

export type HouseOptions = {
  /** Window and door faces on the +z side (the street side). */
  windows?: boolean;
};

/** A house: soft walls `w` (along x) × `d` (along z) × `h`, gable roof rising `rise`. */
export function makeHouse(M: Materials, w: number, d: number, h: number, rise: number, opts: HouseOptions = {}): THREE.Group {
  const grp = new THREE.Group();
  grp.add(softBox(w, h, d, 0.18, M.wall));
  const gs = new THREE.Shape();
  gs.moveTo(-d / 2 - 0.35, 0);
  gs.lineTo(d / 2 + 0.35, 0);
  gs.lineTo(0, rise);
  gs.closePath();
  const rg = new THREE.ExtrudeGeometry(gs, { depth: w + 0.5, bevelEnabled: true, bevelThickness: 0.08, bevelSize: 0.06, bevelSegments: 3 });
  rg.translate(0, 0, -(w + 0.5) / 2);
  rg.rotateY(Math.PI / 2);
  const roof = new THREE.Mesh(rg, M.roof);
  roof.position.y = h;
  grp.add(roof);
  if (opts.windows) {
    const winG = new THREE.BoxGeometry(1.7, 1.1, 0.06);
    const n = Math.max(1, Math.floor(w / 3.2));
    for (let i = 0; i < n; i++) {
      const x = -w / 2 + (i + 0.5) * (w / n);
      if (n > 2 && i === Math.floor(n / 2)) {
        const door = new THREE.Mesh(new THREE.BoxGeometry(1.0, 2.1, 0.06), M.window);
        door.position.set(x, 1.05, d / 2 + 0.01);
        grp.add(door);
      } else {
        const win = new THREE.Mesh(winG, M.window);
        win.position.set(x, Math.min(1.7, h - 0.9), d / 2 + 0.01);
        grp.add(win);
      }
    }
  }
  blob(grp, w * 1.35, d * 1.35, 0.75);
  return grp;
}

/** Smallest-area rectangle around ground points (x, z): centre, long-axis unit vector, half sizes. */
export function minAreaRect(pts: [number, number][]) {
  let best = { area: Infinity, c: [0, 0] as [number, number], u: [1, 0] as [number, number], hl: 0, hs: 0 };
  for (let i = 0; i < pts.length; i++) {
    const [a, b] = [pts[i], pts[(i + 1) % pts.length]];
    const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
    if (len < 1e-6) continue;
    const u: [number, number] = [(b[0] - a[0]) / len, (b[1] - a[1]) / len];
    let [s0, s1, t0, t1] = [Infinity, -Infinity, Infinity, -Infinity];
    for (const p of pts) {
      const s = p[0] * u[0] + p[1] * u[1];
      const t = -p[0] * u[1] + p[1] * u[0];
      [s0, s1, t0, t1] = [Math.min(s0, s), Math.max(s1, s), Math.min(t0, t), Math.max(t1, t)];
    }
    const area = (s1 - s0) * (t1 - t0);
    if (area < best.area) {
      const sc = (s0 + s1) / 2,
        tc = (t0 + t1) / 2;
      const c: [number, number] = [sc * u[0] - tc * u[1], sc * u[1] + tc * u[0]];
      const [hs, ht] = [(s1 - s0) / 2, (t1 - t0) / 2];
      // Long axis first.
      best = hs >= ht ? { area, c, u, hl: hs, hs: ht } : { area, c, u: [-u[1], u[0]], hl: ht, hs };
    }
  }
  return best;
}

function polygonArea(pts: [number, number][]): number {
  let a = 0;
  for (let i = 0; i < pts.length; i++) {
    const [p, q] = [pts[i], pts[(i + 1) % pts.length]];
    a += p[0] * q[1] - q[0] * p[1];
  }
  return Math.abs(a) / 2;
}

/** Hipped roof over a rectangle: eaves at y = 0 (with overhang), ridge `rise` up along the long axis. */
function hipRoof(r: ReturnType<typeof minAreaRect>, rise: number, overhang: number): THREE.BufferGeometry {
  const [cx, cz] = r.c;
  const [ux, uz] = r.u;
  const [vx, vz] = [-uz, ux];
  const hl = r.hl + overhang,
    hs = r.hs + overhang;
  const at = (s: number, t: number, y: number) => new THREE.Vector3(cx + s * ux + t * vx, y, cz + s * uz + t * vz);
  // Ridge inset by half the short side at each end (45°-ish hips scaled by rise).
  const ridge = Math.max(0, hl - hs);
  const e = [at(-hl, -hs, 0), at(hl, -hs, 0), at(hl, hs, 0), at(-hl, hs, 0)];
  const r0 = at(-ridge, 0, rise),
    r1 = at(ridge, 0, rise);
  const tris: THREE.Vector3[] = [];
  const tri = (a: THREE.Vector3, b: THREE.Vector3, c: THREE.Vector3) => tris.push(a, b, c);
  // Long sides (trapezoids), hip ends (triangles). Wound so normals face out and up.
  tri(e[0], r0, e[1]);
  tri(e[1], r0, r1);
  tri(e[2], r1, e[3]);
  tri(e[3], r1, r0);
  tri(e[1], r1, e[2]);
  tri(e[3], r0, e[0]);
  const g = new THREE.BufferGeometry().setFromPoints(tris);
  g.computeVertexNormals();
  // Make sure facets face up (winding depends on the rectangle's handedness).
  const n = g.attributes.normal as THREE.BufferAttribute;
  if (n.getY(0) < 0) {
    const pos = g.attributes.position as THREE.BufferAttribute;
    for (let i = 0; i < pos.count; i += 3) {
      const [x1, y1, z1] = [pos.getX(i + 1), pos.getY(i + 1), pos.getZ(i + 1)];
      pos.setXYZ(i + 1, pos.getX(i + 2), pos.getY(i + 2), pos.getZ(i + 2));
      pos.setXYZ(i + 2, x1, y1, z1);
    }
    g.computeVertexNormals();
  }
  return g;
}

/**
 * A house from a traced outline (ground x, z corners in this object's frame): walls of height
 * `h` following the outline, and a roof — hipped over the outline's tightest rectangle when the
 * house is roughly rectangular, otherwise a flat slab with a soft edge.
 */
export function makeHouseFromOutline(M: Materials, outline: [number, number][], h: number, rise = 1.8): THREE.Group {
  const grp = new THREE.Group();
  if (outline.length < 3) return grp;
  const shape = new THREE.Shape(outline.map(([x, z]) => new THREE.Vector2(x, -z)));
  const walls = new THREE.ExtrudeGeometry(shape, { depth: h, bevelEnabled: false });
  walls.rotateX(-Math.PI / 2);
  grp.add(new THREE.Mesh(walls, M.wall));
  const rect = minAreaRect(outline);
  const rectangular = polygonArea(outline) / (4 * rect.hl * rect.hs) > 0.8;
  if (rectangular) {
    const roof = new THREE.Mesh(hipRoof(rect, Math.min(rise, rect.hs * 0.9), 0.35), M.roof);
    roof.position.y = h;
    grp.add(roof);
  } else {
    const slab = new THREE.ExtrudeGeometry(shape, { depth: 0.12, bevelEnabled: true, bevelThickness: 0.08, bevelSize: 0.3, bevelSegments: 3 });
    slab.rotateX(-Math.PI / 2);
    const roof = new THREE.Mesh(slab, M.roof);
    roof.position.y = h + 0.08;
    grp.add(roof);
  }
  // Soft contact shadow under the house.
  const [cx, cz] = rect.c;
  const b = blob(grp, (rect.hl * 2 + 1) * 1.3, (rect.hs * 2 + 1) * 1.3, 0.6);
  b.position.set(cx, 0.012, cz);
  b.rotation.z = Math.atan2(-rect.u[1], rect.u[0]); // plane laid flat: local x → world x, local y → world −z
  return grp;
}

/** A wheelie bin (~0.6 × 0.7 × 1.05 m): tapered soft body, a slightly wider lid, two wheels at the back. */
export function makeBin(M: Materials): THREE.Group {
  const g = new THREE.Group();
  const body = softBox(0.56, 0.95, 0.66, 0.06, M.trim);
  // Taper: narrower at the bottom.
  shapeVerts(body.geometry, (v) => {
    const k = 0.88 + 0.12 * THREE.MathUtils.clamp(v.y / 0.95, 0, 1);
    v.x *= k;
    v.z *= k;
  });
  g.add(body);
  const lid = softBox(0.62, 0.07, 0.74, 0.03, M.roof);
  lid.position.set(0, 0.95, 0.02);
  g.add(lid);
  const wheelG = new THREE.CylinderGeometry(0.1, 0.1, 0.06, 20);
  wheelG.rotateZ(Math.PI / 2);
  for (const x of [-0.27, 0.27]) {
    const w = new THREE.Mesh(wheelG, M.tyre);
    w.position.set(x, 0.1, -0.3);
    g.add(w);
  }
  blob(g, 1.0, 1.1, 0.6);
  return g;
}

/** A camera on a slim pole: white head, `h` metres up. */
export function makeCameraMount(M: Materials, h: number): THREE.Group {
  const g = new THREE.Group();
  const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.06, h, 12), M.pole);
  pole.position.y = h / 2;
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.17, 24, 16), M.camBody);
  head.position.y = h;
  g.add(pole, head);
  return g;
}
