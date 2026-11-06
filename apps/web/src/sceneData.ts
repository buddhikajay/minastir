import { useEffect, useState } from "react";
import type { Camera } from "./api";

/**
 * What a calibrated camera sees, as a clean top-down map in the camera's own levelled frame
 * (camera at the origin, heading 0): ground types on a fine grid with cars removed, plus columns
 * for walls, fences, trees and poles. Views place it with the camera's current pose, so it follows
 * the camera while you drag it on the plan. All the analysis happens on the server; here we only
 * paint the grid into a canvas for use as a texture.
 */
export type GroundMap = {
  /** Map covers local x, z ∈ [-half, half]. */
  half: number;
  /** Grid cell size, metres. Cell (c, r) spans x = -half + c·cell, z = -half + r·cell. */
  cell: number;
  size: number;
  /** Ground colours, one pixel per cell, row r = z index; transparent where unknown. */
  canvas: HTMLCanvasElement;
};


/** Server label codes (crates/vision/src/segment.rs `Label`). */
export const LABELS = ["other", "road", "sidewalk", "grass", "ground", "building", "tree", "plant", "fence", "vehicle", "sky", "water", "pole"] as const;

/** Muted, light-theme colours per label (RGB 0–255). */
export const LABEL_RGB: [number, number, number][] = [
  [180, 180, 180], // other
  [168, 172, 179], // road
  [222, 212, 194], // sidewalk / footpath / driveway
  [170, 208, 156], // grass
  [208, 192, 162], // bare ground
  [226, 222, 216], // building
  [96, 150, 108], // tree
  [140, 190, 120], // plant
  [214, 190, 150], // fence
  [120, 150, 214], // vehicle
  [190, 220, 240], // sky
  [140, 180, 222], // water
  [130, 132, 136], // pole
];

const cache = new Map<string, Promise<GroundMap | null>>();

function paint(size: number, rle: number[]): HTMLCanvasElement {
  const cv = document.createElement("canvas");
  cv.width = size;
  cv.height = size;
  const ctx = cv.getContext("2d")!;
  const img = ctx.createImageData(size, size);
  const d = img.data;
  let o = 0;
  for (let i = 0; i < rle.length; i += 2) {
    const l = rle[i];
    const n = rle[i + 1];
    if (l === 0) {
      o += n * 4;
      continue;
    }
    const c = LABEL_RGB[l] ?? LABEL_RGB[0];
    for (let k = 0; k < n; k++, o += 4) {
      d[o] = c[0];
      d[o + 1] = c[1];
      d[o + 2] = c[2];
      d[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return cv;
}

function calKey(c: Camera) {
  return `${c.id}:${c.calibration?.computed_at}:${c.calibration?.scale}:${c.pose?.pitch}`;
}

function load(c: Camera): Promise<GroundMap | null> {
  const key = calKey(c);
  let p = cache.get(key);
  if (!p) {
    p = fetch(`/api/cameras/${c.id}/groundmap`)
      .then((r) => (r.ok ? r.json() : null))
      .then((j: { half: number; cell: number; size: number; rle: number[] } | null) => {
        if (!j) return null;
        return { half: j.half, cell: j.cell, size: j.size, canvas: paint(j.size, j.rle) };
      })
      .catch(() => null);
    cache.set(key, p);
  }
  return p;
}

/** Ground maps for every calibrated camera, keyed by camera id. */
export function useGroundMaps(cameras: Camera[]): Record<string, GroundMap> {
  const [maps, setMaps] = useState<Record<string, GroundMap>>({});
  const key = cameras.map(calKey).join("|");
  useEffect(() => {
    let alive = true;
    Promise.all(cameras.filter((c) => c.calibration).map(async (c) => [c.id, await load(c)] as const)).then((pairs) => {
      if (!alive) return;
      const out: Record<string, GroundMap> = {};
      for (const [id, s] of pairs) if (s) out[id] = s;
      setMaps(out);
    });
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  return maps;
}

/** The street as clean parallel bands, fitted on the server from every camera's ground map. */
export type Street = {
  /** Point on the reference line (world x, z). */
  origin: [number, number];
  /** Along-street direction, degrees, yaw convention (0 = +z, 90 = +x). */
  heading: number;
  bands: { t0: number; t1: number; s0: number; s1: number; label: (typeof LABELS)[number]; support: number }[];
  /** Paved strips crossing the verge from the kerb into a lot (street coordinates). */
  driveways: { t0: number; t1: number; s0: number; s1: number }[];
};

/** Driveway paving colour (RGB 0–255). */
export const DRIVEWAY_RGB: [number, number, number] = [196, 197, 199];

/** Walls, trees and poles in world coordinates, merged across cameras. */
export type WorldShapes = {
  /** Buildings get a block of `depth` metres in direction `back` (unit world x, z). */
  walls: { a: [number, number]; b: [number, number]; top: number; label: "building" | "fence"; depth: number; back: [number, number] }[];
  trees: { x: number; z: number; radius: number; top: number; shrub: boolean }[];
  poles: { x: number; z: number; top: number }[];
  bins: { x: number; z: number }[];
};

export type World = { street: Street | null; shapes: WorldShapes };

/** The scene built on the server from every camera (street + merged shapes), refetched whenever
 *  a camera's saved pose or calibration changes. */
export function useWorld(cameras: Camera[], extraKey = ""): World | null {
  const [world, setWorld] = useState<World | null>(null);
  const key = cameras.map((c) => `${calKey(c)}:${c.pose ? [c.pose.x, c.pose.z, c.pose.yaw, c.pose.height].join(",") : "-"}`).join("|") + extraKey;
  useEffect(() => {
    let alive = true;
    fetch("/api/world")
      .then((r) => (r.ok ? r.json() : null))
      .then((w: World | null) => alive && setWorld(w))
      .catch(() => alive && setWorld(null));
    return () => {
      alive = false;
    };
  }, [key]);
  return world;
}

/** World (x, z) of a point given in street coordinates (s along, t across). */
export function streetPoint(st: Street, s: number, t: number): [number, number] {
  const h = (st.heading * Math.PI) / 180;
  return [st.origin[0] + s * Math.sin(h) + t * Math.cos(h), st.origin[1] + s * Math.cos(h) - t * Math.sin(h)];
}

/** A map image (e.g. a satellite screenshot) placed under the plan. Image pixel (u, v) →
 *  world (x, z) + scale · R(rotation) · (u, v); rotation turns +u from +x toward +z. */
export type MapImage = { width: number; height: number; content_type: string; x: number; z: number; scale: number; rotation: number };
export type House = { points: [number, number][]; height: number };
export type RoofPoint = { x: number; z: number; include: boolean };
export type Site = { map: MapImage | null; houses: House[] };

export function imgToWorld(m: MapImage, u: number, v: number): [number, number] {
  const r = (m.rotation * Math.PI) / 180;
  return [m.x + m.scale * (u * Math.cos(r) - v * Math.sin(r)), m.z + m.scale * (u * Math.sin(r) + v * Math.cos(r))];
}

export function worldToImg(m: MapImage, x: number, z: number): [number, number] {
  const r = (m.rotation * Math.PI) / 180;
  const dx = (x - m.x) / m.scale;
  const dz = (z - m.z) / m.scale;
  return [dx * Math.cos(r) + dz * Math.sin(r), -dx * Math.sin(r) + dz * Math.cos(r)];
}

/** Placement that takes image points p1, p2 onto world points w1, w2 (scale, turn, shift). */
export function alignTwoPoints(m: MapImage, p1: [number, number], p2: [number, number], w1: [number, number], w2: [number, number]): MapImage {
  const di = [p2[0] - p1[0], p2[1] - p1[1]];
  const dw = [w2[0] - w1[0], w2[1] - w1[1]];
  const li = Math.hypot(di[0], di[1]);
  if (li < 1e-6) return m;
  const scale = Math.hypot(dw[0], dw[1]) / li;
  const r = Math.atan2(dw[1], dw[0]) - Math.atan2(di[1], di[0]);
  const x = w1[0] - scale * (p1[0] * Math.cos(r) - p1[1] * Math.sin(r));
  const z = w1[1] - scale * (p1[0] * Math.sin(r) + p1[1] * Math.cos(r));
  return { ...m, x, z, scale, rotation: (r * 180) / Math.PI };
}

/** The site plan (map image + traced houses), kept in sync with the server. */
export function useSite() {
  const [site, setSite] = useState<Site | null>(null);
  const [img, setImg] = useState<HTMLImageElement | null>(null);
  const [rev, setRev] = useState(0);
  useEffect(() => {
    fetch("/api/site")
      .then((r) => (r.ok ? r.json() : null))
      .then((s: Site | null) => s && setSite(s))
      .catch(() => {});
  }, []);
  const hasMap = !!site?.map;
  useEffect(() => {
    if (!hasMap) return setImg(null);
    const im = new Image();
    im.onload = () => setImg(im);
    im.src = `/api/site/map?r=${rev}`;
  }, [hasMap, rev]);
  const save = async (next: Site) => {
    setSite(next);
    const r = await fetch("/api/site", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(next) });
    if (r.ok) setSite(await r.json());
  };
  const upload = async (file: File) => {
    const r = await fetch("/api/site/map", { method: "POST", body: file });
    if (!r.ok) throw new Error((await r.json().catch(() => null))?.error ?? "upload failed");
    setSite(await r.json());
    setRev((n) => n + 1);
  };
  const removeMap = async () => {
    const r = await fetch("/api/site/map", { method: "DELETE" });
    if (r.ok) setSite(await r.json());
  };
  /** Let the server place the photo by matching it to the cameras' map. */
  const autoAlign = async (): Promise<{ previous: MapImage; confidence: number }> => {
    const r = await fetch("/api/site/map/align", { method: "POST" });
    const j = await r.json().catch(() => null);
    if (!r.ok) throw new Error(j?.error ?? "auto-align failed");
    setSite(j.site);
    return { previous: j.previous, confidence: j.confidence };
  };
  /** Outline of the roof at the first point, refined by more include / exclude points
   *  (world coordinates; not saved). */
  const findRoof = async (points: RoofPoint[]): Promise<[number, number][]> => {
    const r = await fetch("/api/site/roof", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ points }) });
    const j = await r.json().catch(() => null);
    if (!r.ok) throw new Error(j?.error ?? "couldn't find a roof there");
    return j.points;
  };
  return { site, img, save, upload, removeMap, autoAlign, findRoof };
}
