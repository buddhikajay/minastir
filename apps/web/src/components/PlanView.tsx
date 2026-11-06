import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { Hexagon, ImagePlus, Maximize2, MousePointerClick, Move, Sparkles, Trash2, Undo2 } from "lucide-react";
import { bestProfile, type Camera, type LiveObject, type Pose } from "../api";
import { normYaw } from "./Scene";
import { alignTwoPoints, DRIVEWAY_RGB, imgToWorld, LABELS, LABEL_RGB, streetPoint, worldToImg, type GroundMap, type MapImage, type RoofPoint, type Site, type World } from "../sceneData";

/**
 * Top-down floor plan for arranging cameras by direct manipulation — no modes:
 *   drag a camera          → move it
 *   drag its view (wedge)  → turn it   (Shift snaps to 15°)
 *   drag empty space       → pan;  scroll → zoom
 *   arrows / Q E           → nudge / turn the selected camera
 *
 * Same frame as the 3D scene seen from above: screen-right = +X, screen-down = +Z,
 * so a camera with yaw θ looks along (sin θ, cos θ) on screen. Screen-up is treated as north.
 */

const ACCENT = "#006fff";
const IDLE = "#7d8691";
const rad = (d: number) => (d * Math.PI) / 180;
const deg = (r: number) => (r * 180) / Math.PI;
const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));
const round1 = (v: number) => Math.round(v * 10) / 10;

/** Horizontal field of view guessed from the stream's shape; dual-lens panoramas are ~180°. */
export function hfovOf(c: Camera): number {
  if (c.calibration) {
    // Measured. Lenses of a dual-lens camera meet at the seam, so their widths add up.
    return Math.min(
      359,
      c.calibration.lenses.reduce((a, l) => a + l.hfov, 0),
    );
  }
  const p = bestProfile(c);
  const aspect = p?.width && p?.height ? p.width / p.height : 16 / 9;
  if (aspect >= 2.4) return 180;
  return Math.min(120, deg(2 * Math.atan(aspect * Math.tan(rad(27)))));
}

/** Where the centre of the view meets the ground — the tilt, made visible on the plan. */
export function groundReach(p: Pose): number {
  return p.pitch > 2 ? clamp(p.height / Math.tan(rad(p.pitch)), 1.5, 15) : 15;
}

/** Compass heading (clockwise from screen-up/north). */
export function heading(yaw: number): number {
  return normYaw(180 - yaw);
}

type View = { cx: number; cz: number; scale: number };

/** Same photo, rescaled and/or turned about its centre. */
function aboutCentre(m: MapImage, scale: number, rotation: number): MapImage {
  const [cx, cz] = imgToWorld(m, m.width / 2, m.height / 2);
  const r = rad(rotation);
  const [hu, hv] = [m.width / 2, m.height / 2];
  return { ...m, scale, rotation, x: cx - scale * (hu * Math.cos(r) - hv * Math.sin(r)), z: cz - scale * (hu * Math.sin(r) + hv * Math.cos(r)) };
}
type Mode = "none" | "align" | "trace" | "roof";
type Drag =
  | { kind: "photo"; x: number; y: number; from: MapImage; moved: boolean }
  | { kind: "photoScale"; from: MapImage; start: number; moved: boolean }
  | { kind: "photoRotate"; from: MapImage; start: number; moved: boolean }
  | { kind: "move"; id: string; dx: number; dz: number; moved: boolean }
  | { kind: "rotate"; id: string; moved: boolean }
  | { kind: "pan"; x: number; y: number; from: View; moved: boolean }
  /** Corner of a house outline: the roof preview or saved house `target`. */
  | { kind: "vertex"; target: "preview" | number; index: number; points: [number, number][]; moved: boolean };

export function PlanView({
  cameras,
  poses,
  selected,
  onSelect,
  onPoseChange,
  onPoseCommit,
  objects,
  scenes,
  world,
  site,
  siteImg,
  onSiteSave,
  onUploadMap,
  onRemoveMap,
  onAutoAlign,
  onFindRoof,
}: {
  cameras: Camera[];
  poses: Record<string, Pose>;
  selected: string | null;
  onSelect: (id: string | null) => void;
  /** Live update while dragging (not saved). */
  onPoseChange: (id: string, p: Pose) => void;
  /** Final pose — saved. */
  onPoseCommit: (id: string, p: Pose) => void;
  /** Live tracked objects (from the server) to draw on the plan. */
  objects?: LiveObject[];
  /** What calibrated cameras see, drawn underneath and moved with each camera's pose. */
  scenes?: Record<string, GroundMap>;
  world?: World | null;
  /** Site plan: map image placement and traced houses. */
  site?: Site | null;
  siteImg?: HTMLImageElement | null;
  onSiteSave?: (s: Site) => void;
  onUploadMap?: (f: File) => Promise<void>;
  onRemoveMap?: () => void;
  onAutoAlign?: () => Promise<{ previous: MapImage; confidence: number }>;
  onFindRoof?: (points: RoofPoint[]) => Promise<[number, number][]>;
}) {
  const underlay = useRef<HTMLCanvasElement>(null);
  const wrap = useRef<HTMLDivElement>(null);
  const svg = useRef<SVGSVGElement>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const [view, setView] = useState<View | null>(null);
  const [dragging, setDragging] = useState<Drag["kind"] | null>(null);
  const drag = useRef<Drag | null>(null);
  const latest = useRef<Pose | null>(null);
  const posesRef = useRef(poses);
  posesRef.current = poses;
  // Site tools: align the map image (two points, or drag it), trace house outlines.
  const [mode, setMode] = useState<Mode>("none");
  const [pairs, setPairs] = useState<{ img: [number, number]; world?: [number, number] }[]>([]);
  const [trace, setTrace] = useState<[number, number][]>([]);
  const [hover, setHover] = useState<[number, number] | null>(null);
  const [photoDraft, setPhotoDraft] = useState<MapImage | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [photoOpacity, setPhotoOpacity] = useState(() => {
    try {
      return Number(localStorage.getItem("minastir.photoOpacity")) || 0.9;
    } catch {
      return 0.9;
    }
  });
  useEffect(() => {
    try {
      localStorage.setItem("minastir.photoOpacity", String(photoOpacity));
    } catch {
      /* private mode */
    }
  }, [photoOpacity]);
  const fileInput = useRef<HTMLInputElement>(null);
  const [aligning, setAligning] = useState(false);
  const [finding, setFinding] = useState(false);
  // Click-a-roof: prompt points and the outline they give (not saved until accepted).
  const [roofPts, setRoofPts] = useState<RoofPoint[]>([]);
  const [roofPreview, setRoofPreview] = useState<[number, number][] | null>(null);
  // Outline being edited (corner drag), shown live and saved on release.
  const [polyDraft, setPolyDraft] = useState<{ target: "preview" | number; points: [number, number][] } | null>(null);
  const [selHouse, setSelHouse] = useState<number | null>(null);
  const [confirmClear, setConfirmClear] = useState(false);
  useEffect(() => {
    if (!confirmClear) return;
    const t = setTimeout(() => setConfirmClear(false), 3000);
    return () => clearTimeout(t);
  }, [confirmClear]);
  const [undoMap, setUndoMap] = useState<MapImage | null>(null);
  const runAutoAlign = async () => {
    if (!onAutoAlign) return;
    setAligning(true);
    setNotice("Matching the photo to the cameras' map…");
    try {
      const { previous, confidence } = await onAutoAlign();
      setUndoMap(previous);
      setNotice(
        confidence >= 1.1
          ? "Photo aligned automatically — check the road edges and fine-tune if needed"
          : "Photo aligned, but the match is uncertain — check it, or align by hand",
      );
    } catch (err) {
      setNotice(String((err as Error).message ?? err));
    } finally {
      setAligning(false);
    }
  };
  const map = photoDraft ?? site?.map ?? null;
  const switchMode = (m: Mode) => {
    setMode((cur) => (cur === m ? "none" : m));
    setPairs([]);
    setTrace([]);
    setRoofPts([]);
    setRoofPreview(null);
    setSelHouse(null);
    setNotice(null);
  };

  useLayoutEffect(() => {
    const el = wrap.current!;
    const ro = new ResizeObserver(([e]) => setSize({ w: e.contentRect.width, h: e.contentRect.height }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const fit = useCallback(() => {
    if (!size.w) return;
    const ps = Object.values(posesRef.current);
    if (!ps.length) return setView({ cx: 0, cz: 0, scale: 30 });
    // Include each camera's reach so its view stays on screen.
    const xs = ps.flatMap((p) => [p.x, p.x + Math.sin(rad(p.yaw)) * groundReach(p)]);
    const zs = ps.flatMap((p) => [p.z, p.z + Math.cos(rad(p.yaw)) * groundReach(p)]);
    const [x0, x1, z0, z1] = [Math.min(...xs), Math.max(...xs), Math.min(...zs), Math.max(...zs)];
    const pad = 2.5;
    const scale = clamp(Math.min(size.w / (x1 - x0 + 2 * pad), size.h / (z1 - z0 + 2 * pad)), 8, 160);
    setView({ cx: (x0 + x1) / 2, cz: (z0 + z1) / 2, scale });
  }, [size.w, size.h]);

  useEffect(() => {
    if (!view && size.w) fit();
  }, [view, size.w, fit]);

  // Wheel zoom around the cursor (non-passive so the page doesn't scroll).
  useEffect(() => {
    const el = wrap.current!;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = el.getBoundingClientRect();
      const mx = e.clientX - r.left - r.width / 2;
      const my = e.clientY - r.top - r.height / 2;
      setView((v) => {
        if (!v) return v;
        // 0.8 px/m shows the whole neighbourhood (~1.5 km across); 250 px/m is close-up.
        const scale = clamp(v.scale * Math.exp(-e.deltaY * 0.0015), 0.8, 250);
        const wx = v.cx + mx / v.scale;
        const wz = v.cz + my / v.scale;
        return { scale, cx: wx - mx / scale, cz: wz - my / scale };
      });
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  // Keyboard nudging for the selected camera.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (t?.tagName === "INPUT" || t?.tagName === "TEXTAREA") return;
      if (mode !== "none") {
        if (e.key === "Escape") {
          if (trace.length || pairs.length || roofPts.length) {
            setTrace([]);
            setPairs([]);
            setRoofPts([]);
            setRoofPreview(null);
            setNotice(null);
          } else setMode("none");
        } else if (e.key === "Enter" && mode === "roof") {
          acceptRoof();
        } else if ((e.key === "Delete" || e.key === "Backspace") && selHouse !== null && !trace.length) {
          deleteHouse(selHouse);
        } else if (e.key === "Backspace" && mode === "trace") {
          setTrace((t) => t.slice(0, -1));
        } else if (e.key === "Enter" && mode === "trace") {
          closeHouse();
        } else if (mode === "align" && map && site) {
          // Fine adjustment: arrows move, +/− scale, Q/E turn (Shift = bigger steps).
          const big = e.shiftKey;
          const step = big ? 1 : 0.1;
          const k = e.key.toLowerCase();
          let next: MapImage | null = null;
          if (e.key === "ArrowLeft") next = { ...map, x: map.x - step };
          else if (e.key === "ArrowRight") next = { ...map, x: map.x + step };
          else if (e.key === "ArrowUp") next = { ...map, z: map.z - step };
          else if (e.key === "ArrowDown") next = { ...map, z: map.z + step };
          else if (k === "+" || k === "=") next = aboutCentre(map, map.scale * (big ? 1.05 : 1.005), map.rotation);
          else if (k === "-" || k === "_") next = aboutCentre(map, map.scale / (big ? 1.05 : 1.005), map.rotation);
          else if (k === "q") next = aboutCentre(map, map.scale, map.rotation - (big ? 2 : 0.2));
          else if (k === "e") next = aboutCentre(map, map.scale, map.rotation + (big ? 2 : 0.2));
          else return;
          onSiteSave?.({ ...site, map: next });
        } else return;
        e.preventDefault();
        return;
      }
      if (!selected) return;
      const p = posesRef.current[selected];
      if (!p) return;
      const step = e.shiftKey ? 1 : 0.1;
      const turn = e.shiftKey ? 15 : 5;
      const next: Partial<Record<string, Pose>> = {
        ArrowLeft: { ...p, x: round1(p.x - step) },
        ArrowRight: { ...p, x: round1(p.x + step) },
        ArrowUp: { ...p, z: round1(p.z - step) },
        ArrowDown: { ...p, z: round1(p.z + step) },
        // On screen, Q turns anticlockwise and E clockwise (yaw grows anticlockwise from above).
        q: { ...p, yaw: normYaw(p.yaw + turn) },
        e: { ...p, yaw: normYaw(p.yaw - turn) },
      };
      const n = next[e.key.length === 1 ? e.key.toLowerCase() : e.key];
      if (!n) return;
      e.preventDefault();
      onPoseCommit(selected, n);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected, onPoseCommit, mode, trace, pairs, map, site, roofPts, roofPreview, selHouse]);

  const v = view ?? { cx: 0, cz: 0, scale: 30 };
  const sx = (x: number) => size.w / 2 + (x - v.cx) * v.scale;
  const sy = (z: number) => size.h / 2 + (z - v.cz) * v.scale;
  const toWorld = (clientX: number, clientY: number) => {
    const r = svg.current!.getBoundingClientRect();
    return { x: v.cx + (clientX - r.left - size.w / 2) / v.scale, z: v.cz + (clientY - r.top - size.h / 2) / v.scale };
  };

  function begin(e: React.PointerEvent, d: Drag) {
    e.stopPropagation();
    drag.current = d;
    latest.current = null;
    svg.current!.setPointerCapture(e.pointerId);
    setDragging(d.kind);
  }

  function onMove(e: React.PointerEvent) {
    if (mode !== "none" && svg.current) {
      const w = toWorld(e.clientX, e.clientY);
      setHover([w.x, w.z]);
    }
    const d = drag.current;
    if (!d) return;
    if (d.kind === "photo") {
      const dx = e.clientX - d.x;
      const dy = e.clientY - d.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) d.moved = true;
      if (d.moved) setPhotoDraft({ ...d.from, x: d.from.x + dx / v.scale, z: d.from.z + dy / v.scale });
      return;
    }
    if (d.kind === "vertex") {
      const w = toWorld(e.clientX, e.clientY);
      const pts = d.points.slice();
      pts[d.index] = [Math.round(w.x * 100) / 100, Math.round(w.z * 100) / 100];
      d.points = pts;
      d.moved = true;
      setPolyDraft({ target: d.target, points: pts });
      return;
    }
    if (d.kind === "photoScale" || d.kind === "photoRotate") {
      const [cx, cz] = imgToWorld(d.from, d.from.width / 2, d.from.height / 2);
      const w = toWorld(e.clientX, e.clientY);
      d.moved = true;
      if (d.kind === "photoScale") {
        const k = Math.hypot(w.x - cx, w.z - cz) / Math.max(d.start, 1e-6);
        setPhotoDraft(aboutCentre(d.from, d.from.scale * k, d.from.rotation));
      } else {
        let rot = d.from.rotation + deg(Math.atan2(w.z - cz, w.x - cx)) - d.start;
        if (e.shiftKey) rot = Math.round(rot);
        setPhotoDraft(aboutCentre(d.from, d.from.scale, rot));
      }
      return;
    }
    if (d.kind === "pan") {
      const dx = e.clientX - d.x;
      const dy = e.clientY - d.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) d.moved = true;
      setView({ ...d.from, cx: d.from.cx - dx / d.from.scale, cz: d.from.cz - dy / d.from.scale });
      return;
    }
    const p = posesRef.current[d.id];
    if (!p) return;
    const w = toWorld(e.clientX, e.clientY);
    let next: Pose;
    if (d.kind === "move") {
      next = { ...p, x: round1(w.x - d.dx), z: round1(w.z - d.dz) };
    } else {
      let yaw = deg(Math.atan2(w.x - p.x, w.z - p.z));
      if (e.shiftKey) yaw = Math.round(yaw / 15) * 15;
      next = { ...p, yaw: normYaw(yaw) };
    }
    if (next.x === p.x && next.z === p.z && next.yaw === p.yaw) return;
    d.moved = true;
    latest.current = next;
    onPoseChange(d.id, next);
  }

  function onUp(e: React.PointerEvent) {
    const d = drag.current;
    drag.current = null;
    setDragging(null);
    if (svg.current?.hasPointerCapture(e.pointerId)) svg.current.releasePointerCapture(e.pointerId);
    if (!d) return;
    if (d.kind === "vertex") {
      let pts = d.points;
      // Option-click a corner removes it (keeping at least a triangle).
      if (!d.moved && e.altKey && pts.length > 3) pts = pts.filter((_, i) => i !== d.index);
      if (d.moved || pts !== d.points) commitOutline(d.target, pts);
      setPolyDraft(null);
      return;
    }
    if (d.kind === "photo" || d.kind === "photoScale" || d.kind === "photoRotate") {
      if (d.moved && photoDraft && site) onSiteSave?.({ ...site, map: photoDraft });
      else if (!d.moved && d.kind === "photo") siteClick(e);
      setPhotoDraft(null);
    } else if (d.kind === "pan") {
      if (d.moved) return;
      if (selHouse !== null) setSelHouse(null);
      else if (mode === "trace" || mode === "roof") siteClick(e);
      else onSelect(null);
    } else if (d.moved && latest.current) {
      onPoseCommit(d.id, latest.current);
    }
  }

  /** Snap to a camera's position when clicking close to it (handy alignment targets). */
  function snapped(clientX: number, clientY: number): [number, number] {
    const w = toWorld(clientX, clientY);
    for (const p of Object.values(posesRef.current)) {
      if (Math.hypot((p.x - w.x) * v.scale, (p.z - w.z) * v.scale) < 12) return [p.x, p.z];
    }
    return [w.x, w.z];
  }

  function siteClick(e: React.PointerEvent) {
    if (mode === "align" && map && site) {
      const last = pairs[pairs.length - 1];
      if (!last || last.world) {
        const w = toWorld(e.clientX, e.clientY);
        const img = worldToImg(map, w.x, w.z);
        setPairs((ps) => (ps.length >= 2 ? [{ img }] : [...ps, { img }]));
      } else {
        const done = [...pairs.slice(0, -1), { ...last, world: snapped(e.clientX, e.clientY) }];
        if (done.length === 2) {
          const [a, b] = done as { img: [number, number]; world: [number, number] }[];
          const next = alignTwoPoints(map, a.img, b.img, a.world, b.world);
          onSiteSave?.({ ...site, map: next });
          setNotice(`Photo aligned · ${(1 / next.scale).toFixed(1)} px per metre`);
          setPairs([]);
        } else setPairs(done);
      }
    } else if (mode === "roof" && site && onFindRoof && !finding) {
      const w = toWorld(e.clientX, e.clientY);
      const refine = (e.shiftKey || e.altKey) && roofPts.length > 0;
      let pts: RoofPoint[];
      if (refine) {
        pts = [...roofPts, { x: w.x, z: w.z, include: !e.altKey }];
      } else {
        acceptRoof(); // a plain click on the next roof keeps the current outline
        pts = [{ x: w.x, z: w.z, include: true }];
      }
      setRoofPts(pts);
      setFinding(true);
      setNotice(refine ? "Refining…" : "Finding the roof…");
      onFindRoof(pts)
        .then((points) => {
          setRoofPreview(points);
          setNotice("Drag corners to adjust · Shift-click to add a missed part · Option-click to remove one · Enter (or click the next roof) to keep · Esc to discard");
        })
        .catch((err) => {
          setNotice(String((err as Error).message ?? err));
          if (!refine) setRoofPts([]);
        })
        .finally(() => setFinding(false));
    } else if (mode === "trace" && site) {
      const pt = snapped(e.clientX, e.clientY);
      if (trace.length >= 3 && Math.hypot((pt[0] - trace[0][0]) * v.scale, (pt[1] - trace[0][1]) * v.scale) < 12) {
        closeHouse();
      } else setTrace((t) => [...t, pt]);
    }
  }

  function deleteHouse(i: number) {
    if (!site) return;
    onSiteSave?.({ ...site, houses: site.houses.filter((_, k) => k !== i) });
    setSelHouse(null);
    setNotice("House removed");
  }

  function commitOutline(target: "preview" | number, points: [number, number][]) {
    if (target === "preview") setRoofPreview(points);
    else if (site) onSiteSave?.({ ...site, houses: site.houses.map((h, i) => (i === target ? { ...h, points } : h)) });
  }

  function acceptRoof() {
    if (!site || !roofPreview) return;
    onSiteSave?.({ ...site, houses: [...site.houses, { points: roofPreview, height: 4.5 }] });
    setRoofPreview(null);
    setRoofPts([]);
    setNotice("House added — drag its corners to adjust, or click another roof");
  }

  function closeHouse() {
    if (!site || trace.length < 3) return;
    onSiteSave?.({ ...site, houses: [...site.houses, { points: trace, height: 4.5 }] });
    setTrace([]);
    setNotice("House added");
  }

  // ----- grid -----
  const grid = useMemo(() => {
    if (!size.w) return { minor: "", major: "" };
    // Grid spacing that stays readable at any zoom: 1 / 5 / 25 / 100 m, major lines every 5 steps.
    const step = v.scale >= 14 ? 1 : v.scale >= 3 ? 5 : v.scale >= 0.9 ? 25 : 100;
    const majorStep = step * 5;
    const x0 = v.cx - size.w / 2 / v.scale;
    const x1 = v.cx + size.w / 2 / v.scale;
    const z0 = v.cz - size.h / 2 / v.scale;
    const z1 = v.cz + size.h / 2 / v.scale;
    let minor = "";
    let major = "";
    for (let x = Math.ceil(x0 / step) * step; x <= x1; x += step) {
      const seg = `M${sx(x)} 0V${size.h}`;
      if (Math.round(x) % majorStep === 0) major += seg;
      else minor += seg;
    }
    for (let z = Math.ceil(z0 / step) * step; z <= z1; z += step) {
      const seg = `M0 ${sy(z)}H${size.w}`;
      if (Math.round(z) % majorStep === 0) major += seg;
      else minor += seg;
    }
    return { minor, major };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [size.w, size.h, v.cx, v.cz, v.scale]);

  const scaleBar = useMemo(() => {
    const m = [1, 2, 5, 10, 20, 50, 100, 200, 500].find((n) => n * v.scale >= 70) ?? 500;
    return { m, px: m * v.scale };
  }, [v.scale]);

  // Scene underlay: every calibrated camera's ground map, placed with its *current* pose, so
  // dragging a camera slides its view of the road/houses over the others' until they line up.
  useEffect(() => {
    const cv = underlay.current;
    if (!cv || !size.w || !size.h) return;
    cv.width = size.w;
    cv.height = size.h;
    const ctx = cv.getContext("2d");
    if (!ctx || !scenes) return;
    ctx.clearRect(0, 0, cv.width, cv.height);
    // Map image (satellite screenshot) underneath everything.
    if (map && siteImg) {
      const r = rad(map.rotation);
      const k = map.scale * v.scale;
      ctx.setTransform(k * Math.cos(r), k * Math.sin(r), -k * Math.sin(r), k * Math.cos(r), sx(map.x), sy(map.z));
      ctx.globalAlpha = photoOpacity;
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(siteImg, 0, 0);
      ctx.setTransform(1, 0, 0, 1, 0, 0);
    }
    const photo = !!(map && siteImg);
    // Ground maps, selected camera last so it sits on top.
    const ordered = [...cameras].sort((a, b) => Number(a.id === selected) - Number(b.id === selected));
    for (const c of ordered) {
      const g = scenes[c.id];
      const p = poses[c.id];
      if (!g || !p) continue;
      const sn = Math.sin(rad(p.yaw));
      const cs = Math.cos(rad(p.yaw));
      const at = (lx: number, lz: number) => [sx(p.x + lx * cs + lz * sn), sy(p.z - lx * sn + lz * cs)];
      const o = at(-g.half, -g.half);
      const a = at(-g.half + g.cell, -g.half);
      const b = at(-g.half, -g.half + g.cell);
      ctx.setTransform(a[0] - o[0], a[1] - o[1], b[0] - o[0], b[1] - o[1], o[0], o[1]);
      ctx.globalAlpha = photo ? (c.id === selected ? 0.55 : 0.25) : c.id === selected ? 0.8 : 0.5;
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(g.canvas, 0, 0);
      ctx.setTransform(1, 0, 0, 1, 0, 0);
    }
    const street = world?.street;
    const poly = (pts: [number, number][]) => {
      ctx.beginPath();
      for (const [wx, wz] of pts) ctx.lineTo(sx(wx), sy(wz));
      ctx.closePath();
      ctx.fill();
    };
    if (street) {
      // Fitted street over the ground maps, then driveways.
      ctx.globalAlpha = photo ? 0.55 : 1;
      for (const b of street.bands) {
        const rgb = LABEL_RGB[LABELS.indexOf(b.label)] ?? LABEL_RGB[0];
        ctx.fillStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
        poly([streetPoint(street, b.s0, b.t0), streetPoint(street, b.s1, b.t0), streetPoint(street, b.s1, b.t1), streetPoint(street, b.s0, b.t1)]);
        if (b.label === "road") {
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = Math.max(1, 0.16 * v.scale);
          for (const t of [b.t0, b.t1]) {
            const [ax, az] = streetPoint(street, b.s0, t);
            const [bx, bz] = streetPoint(street, b.s1, t);
            ctx.beginPath();
            ctx.moveTo(sx(ax), sy(az));
            ctx.lineTo(sx(bx), sy(bz));
            ctx.stroke();
          }
        }
      }
      ctx.fillStyle = `rgb(${DRIVEWAY_RGB.join(",")})`;
      for (const d of street.driveways) {
        poly([streetPoint(street, d.s0, d.t0), streetPoint(street, d.s1, d.t0), streetPoint(street, d.s1, d.t1), streetPoint(street, d.s0, d.t1)]);
      }
    }
    if (world) {
      // House blocks (soft), walls and fences (lines), trees (circles), poles (dots).
      const { walls, trees, poles } = world.shapes;
      ctx.globalAlpha = 0.5;
      ctx.fillStyle = "#dcd9d3";
      for (const w of walls) {
        if (w.depth <= 0) continue;
        const [bx, bz] = [w.back[0] * w.depth, w.back[1] * w.depth];
        poly([w.a, w.b, [w.b[0] + bx, w.b[1] + bz], [w.a[0] + bx, w.a[1] + bz]]);
      }
      ctx.globalAlpha = 0.9;
      ctx.lineCap = "round";
      for (const w of walls) {
        ctx.strokeStyle = w.label === "fence" ? "#b8ad9c" : "#9c958c";
        ctx.lineWidth = Math.max(1, (w.label === "fence" ? 0.1 : 0.25) * v.scale);
        ctx.beginPath();
        ctx.moveTo(sx(w.a[0]), sy(w.a[1]));
        ctx.lineTo(sx(w.b[0]), sy(w.b[1]));
        ctx.stroke();
      }
      ctx.globalAlpha = 0.6;
      for (const t of trees) {
        ctx.fillStyle = t.shrub ? "#bcd8b0" : "#a9cba3";
        ctx.beginPath();
        ctx.arc(sx(t.x), sy(t.z), Math.max(2, t.radius * v.scale), 0, Math.PI * 2);
        ctx.fill();
      }
      // Bins: small rounded squares, ~0.7 m.
      ctx.globalAlpha = 0.9;
      ctx.fillStyle = "#5e636b";
      for (const b of world.shapes.bins ?? []) {
        const s = Math.max(4, 0.7 * v.scale);
        ctx.beginPath();
        ctx.roundRect(sx(b.x) - s / 2, sy(b.z) - s / 2, s, s, Math.min(3, s / 4));
        ctx.fill();
      }
      ctx.globalAlpha = 1;
      ctx.fillStyle = "#8a9096";
      for (const q of poles) {
        ctx.beginPath();
        ctx.arc(sx(q.x), sy(q.z), Math.max(1.5, 0.12 * v.scale), 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.globalAlpha = 1;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scenes, world, map, siteImg, photoOpacity, poses, cameras, selected, size.w, size.h, v.cx, v.cz, v.scale]);

  // Draw the selected camera last so it sits on top.
  const order = [...cameras].sort((a, b) => Number(a.id === selected) - Number(b.id === selected));

  return (
    <div
      ref={wrap}
      className="absolute inset-0 select-none overflow-hidden bg-[#eef0f2]"
      style={{
        cursor:
          mode !== "none" && !dragging
            ? "crosshair"
            : dragging === "pan" || dragging === "rotate" || dragging === "photo" || dragging === "photoRotate" || dragging === "vertex"
              ? "grabbing"
              : dragging === "move"
                ? "move"
                : dragging === "photoScale"
                  ? "nwse-resize"
                : "default",
      }}
    >
      <canvas ref={underlay} className="pointer-events-none absolute inset-0" style={{ width: size.w, height: size.h }} />
      <svg
        ref={svg}
        width={size.w}
        height={size.h}
        className="absolute inset-0 touch-none"
        onPointerDown={(e) =>
          mode === "align" && map
            ? begin(e, { kind: "photo", x: e.clientX, y: e.clientY, from: map, moved: false })
            : begin(e, { kind: "pan", x: e.clientX, y: e.clientY, from: v, moved: false })
        }
        onPointerLeave={() => setHover(null)}
        onPointerMove={onMove}
        onPointerUp={onUp}
        onPointerCancel={onUp}
      >
        <path d={grid.minor} stroke="#e1e4e8" strokeWidth={1} />
        <path d={grid.major} stroke="#cfd3d9" strokeWidth={1} />

        {site?.houses.map((h, i) => {
          const editing = mode === "roof" || mode === "trace";
          const sel = editing && selHouse === i;
          return (
            <polygon
              key={`h${i}`}
              points={(polyDraft?.target === i ? polyDraft.points : h.points).map(([x, z]) => `${sx(x)},${sy(z)}`).join(" ")}
              fill={sel ? "#dbe8ff" : "#e4e1dc"}
              fillOpacity={0.85}
              stroke={sel ? ACCENT : "#8f887e"}
              strokeWidth={sel ? 2 : 1.5}
              pointerEvents={editing && !roofPreview && !trace.length ? "auto" : "none"}
              style={{ cursor: "pointer" }}
              onPointerDown={(e) => {
                e.stopPropagation();
                setSelHouse(i);
                setNotice("Drag corners to adjust · Delete to remove this house");
              }}
            />
          );
        })}
        {mode === "trace" && trace.length > 0 && (
          <g pointerEvents="none">
            <polyline
              points={[...trace, ...(hover ? [hover] : [])].map(([x, z]) => `${sx(x)},${sy(z)}`).join(" ")}
              fill={ACCENT}
              fillOpacity={0.08}
              stroke={ACCENT}
              strokeWidth={1.5}
              strokeDasharray="4 3"
            />
            {trace.map(([x, z], i) => (
              <circle key={i} cx={sx(x)} cy={sy(z)} r={i === 0 ? 6 : 4} fill="#fff" stroke={ACCENT} strokeWidth={1.5} />
            ))}
          </g>
        )}
        {mode === "roof" && (roofPreview || roofPts.length > 0) && (
          <g pointerEvents="none">
            {roofPreview && (
              <polygon
                points={(polyDraft?.target === "preview" ? polyDraft.points : roofPreview).map(([x, z]) => `${sx(x)},${sy(z)}`).join(" ")}
                fill={ACCENT}
                fillOpacity={finding ? 0.06 : 0.14}
                stroke={ACCENT}
                strokeWidth={2}
                strokeDasharray="6 4"
              />
            )}
            {roofPts.map((p, i) => (
              <g key={i} transform={`translate(${sx(p.x)} ${sy(p.z)})`}>
                <circle r={7} fill={p.include ? "#12b76a" : "#f04438"} stroke="#fff" strokeWidth={1.5} />
                <path d={p.include ? "M-3.5 0H3.5M0 -3.5V3.5" : "M-3.5 0H3.5"} stroke="#fff" strokeWidth={1.8} />
              </g>
            ))}
          </g>
        )}
        {(mode === "roof" || mode === "trace") &&
          !finding &&
          [
            ...(site?.houses ?? []).map((h, i) => ({ target: i as "preview" | number, points: polyDraft?.target === i ? polyDraft.points : h.points })),
            ...(roofPreview ? [{ target: "preview" as const, points: polyDraft?.target === "preview" ? polyDraft.points : roofPreview }] : []),
          ].map(({ target, points }) => {
            const color = target === "preview" ? ACCENT : "#6b645b";
            return (
              <g key={`edit-${target}`}>
                {/* Edge midpoints: drag to add a corner. */}
                {points.map((p, i) => {
                  const q = points[(i + 1) % points.length];
                  const m: [number, number] = [(p[0] + q[0]) / 2, (p[1] + q[1]) / 2];
                  if (Math.hypot(q[0] - p[0], q[1] - p[1]) * v.scale < 28) return null;
                  return (
                    <circle
                      key={`m${i}`}
                      cx={sx(m[0])}
                      cy={sy(m[1])}
                      r={3.5}
                      fill="#fff"
                      fillOpacity={0.85}
                      stroke={color}
                      strokeWidth={1}
                      style={{ cursor: "copy" }}
                      onPointerDown={(e) => {
                        const pts = [...points.slice(0, i + 1), m, ...points.slice(i + 1)];
                        begin(e, { kind: "vertex", target, index: i + 1, points: pts, moved: true });
                        setPolyDraft({ target, points: pts });
                      }}
                    />
                  );
                })}
                {/* Corners: drag to move, Option-click to remove. */}
                {points.map((p, i) => (
                  <rect
                    key={`c${i}`}
                    x={sx(p[0]) - 4.5}
                    y={sy(p[1]) - 4.5}
                    width={9}
                    height={9}
                    rx={1.5}
                    fill="#fff"
                    stroke={color}
                    strokeWidth={1.75}
                    style={{ cursor: "move" }}
                    onPointerDown={(e) => begin(e, { kind: "vertex", target, index: i, points: points.slice(), moved: false })}
                  />
                ))}
              </g>
            );
          })}
        {(mode === "roof" || mode === "trace") &&
          selHouse !== null &&
          site?.houses[selHouse] &&
          (() => {
            const pts = site.houses[selHouse].points;
            const c = [pts.reduce((a, p) => a + p[0], 0) / pts.length, pts.reduce((a, p) => a + p[1], 0) / pts.length];
            return (
              <g
                transform={`translate(${sx(c[0])} ${sy(c[1])})`}
                style={{ cursor: "pointer" }}
                onPointerDown={(e) => {
                  e.stopPropagation();
                  deleteHouse(selHouse);
                }}
              >
                <title>Remove this house</title>
                <circle r={12} fill="#f04438" stroke="#fff" strokeWidth={2} filter="drop-shadow(0 1px 2px rgb(16 24 40 / 0.25))" />
                <path d="M-4.5 -4.5L4.5 4.5M4.5 -4.5L-4.5 4.5" stroke="#fff" strokeWidth={2.2} strokeLinecap="round" />
              </g>
            );
          })()}
        {mode === "align" && map && (() => {
          // Frame with resize corners and a rotate handle above the top edge.
          const corners = [imgToWorld(map, 0, 0), imgToWorld(map, map.width, 0), imgToWorld(map, map.width, map.height), imgToWorld(map, 0, map.height)].map(
            ([x, z]) => [sx(x), sy(z)] as const,
          );
          const [cx, cz] = imgToWorld(map, map.width / 2, map.height / 2);
          const topMid = [(corners[0][0] + corners[1][0]) / 2, (corners[0][1] + corners[1][1]) / 2];
          const c = [sx(cx), sy(cz)];
          const out = [topMid[0] - c[0], topMid[1] - c[1]];
          const ol = Math.hypot(out[0], out[1]) || 1;
          const knob = [topMid[0] + (out[0] / ol) * 28, topMid[1] + (out[1] / ol) * 28];
          const startScale = (e: React.PointerEvent) => {
            const w = toWorld(e.clientX, e.clientY);
            begin(e, { kind: "photoScale", from: map, start: Math.hypot(w.x - cx, w.z - cz), moved: false });
          };
          const startRotate = (e: React.PointerEvent) => {
            const w = toWorld(e.clientX, e.clientY);
            begin(e, { kind: "photoRotate", from: map, start: deg(Math.atan2(w.z - cz, w.x - cx)), moved: false });
          };
          return (
            <g>
              <polygon points={corners.map((p) => p.join(",")).join(" ")} fill="none" stroke={ACCENT} strokeWidth={1.5} strokeDasharray="6 4" pointerEvents="none" />
              <line x1={topMid[0]} y1={topMid[1]} x2={knob[0]} y2={knob[1]} stroke={ACCENT} strokeWidth={1.5} pointerEvents="none" />
              <circle cx={knob[0]} cy={knob[1]} r={8} fill="#fff" stroke={ACCENT} strokeWidth={2} style={{ cursor: "grab" }} onPointerDown={startRotate} />
              {corners.map(([x, y], i) => (
                <rect key={i} x={x - 6} y={y - 6} width={12} height={12} rx={2} fill="#fff" stroke={ACCENT} strokeWidth={2} style={{ cursor: "nwse-resize" }} onPointerDown={startScale} />
              ))}
            </g>
          );
        })()}
        {mode === "align" && map && (
          <g pointerEvents="none">
            {pairs.map((pr, i) => {
              const r = rad(map.rotation);
              const ix = map.x + map.scale * (pr.img[0] * Math.cos(r) - pr.img[1] * Math.sin(r));
              const iz = map.z + map.scale * (pr.img[0] * Math.sin(r) + pr.img[1] * Math.cos(r));
              const to = pr.world ?? hover;
              return (
                <g key={i}>
                  {to && <line x1={sx(ix)} y1={sy(iz)} x2={sx(to[0])} y2={sy(to[1])} stroke="#f08c00" strokeWidth={1.5} strokeDasharray="4 3" />}
                  <circle cx={sx(ix)} cy={sy(iz)} r={5} fill="#f08c00" stroke="#fff" strokeWidth={1.5} />
                  {pr.world && <circle cx={sx(pr.world[0])} cy={sy(pr.world[1])} r={5} fill={ACCENT} stroke="#fff" strokeWidth={1.5} />}
                </g>
              );
            })}
          </g>
        )}

        {objects?.map((o) => {
          const x = sx(o.x);
          const y = sy(o.z);
          const vehicle = o.class === "car" || o.class === "truck" || o.class === "bus";
          if (!vehicle) return <circle key={o.id} cx={x} cy={y} r={o.class === "person" ? 5 : 4} fill="#4b5563" pointerEvents="none" />;
          const [w, l] = o.class === "car" ? [1.85, 4.4] : [2.3, 7];
          // Rect drawn along +Y (screen-down = +Z), then rotated to the heading.
          return (
            <rect
              key={o.id}
              x={x - (w * v.scale) / 2}
              y={y - (l * v.scale) / 2}
              width={w * v.scale}
              height={l * v.scale}
              rx={3}
              fill="#4b5563"
              fillOpacity={0.28}
              stroke="#374151"
              strokeOpacity={0.8}
              strokeWidth={1.25}
              transform={`rotate(${-o.heading} ${x} ${y})`}
              pointerEvents="none"
            />
          );
        })}

        {order.map((c) => {
          const p = poses[c.id];
          if (!p) return null;
          const sel = c.id === selected;
          const color = sel ? ACCENT : IDLE;
          const cx = sx(p.x);
          const cy = sy(p.z);
          const r = Math.max(groundReach(p) * v.scale, 56);
          const half = rad(hfovOf(c) / 2);
          const f = rad(p.yaw);
          // Screen direction for an angle offset from the forward vector.
          const dir = (off: number) => [Math.sin(f + off), Math.cos(f + off)] as const;
          const [ax, ay] = dir(-half);
          const [bx, by] = dir(half);
          const [fx, fy] = dir(0);
          const large = half * 2 > Math.PI ? 1 : 0;
          // Angle offset −half→+half runs anticlockwise on screen → sweep-flag 0.
          const wedge = `M${cx} ${cy}L${cx + ax * r} ${cy + ay * r}A${r} ${r} 0 ${large} 0 ${cx + bx * r} ${cy + by * r}Z`;
          const tip = [cx + fx * r, cy + fy * r];
          const rotate = (e: React.PointerEvent) => {
            onSelect(c.id);
            begin(e, { kind: "rotate", id: c.id, moved: false });
          };
          return (
            <g key={c.id} pointerEvents={mode === "none" ? undefined : "none"}>
              <path
                d={wedge}
                fill={color}
                fillOpacity={sel ? 0.13 : 0.07}
                stroke={color}
                strokeOpacity={sel ? 0.55 : 0.3}
                strokeWidth={1}
                style={{ cursor: dragging ? undefined : "grab" }}
                onPointerDown={rotate}
              />
              <line x1={cx} y1={cy} x2={tip[0]} y2={tip[1]} stroke={color} strokeOpacity={sel ? 0.7 : 0.35} strokeDasharray="3 4" pointerEvents="none" />
              {/* rotation handle at the aim point */}
              <circle
                cx={tip[0]}
                cy={tip[1]}
                r={sel ? 7 : 5}
                fill="#fff"
                stroke={color}
                strokeWidth={sel ? 2 : 1.5}
                style={{ cursor: dragging ? undefined : "grab" }}
                onPointerDown={rotate}
              />
              {/* body */}
              <g
                style={{ cursor: dragging ? undefined : "move" }}
                onPointerDown={(e) => {
                  onSelect(c.id);
                  const w = toWorld(e.clientX, e.clientY);
                  begin(e, { kind: "move", id: c.id, dx: w.x - p.x, dz: w.z - p.z, moved: false });
                }}
              >
                <circle cx={cx} cy={cy} r={16} fill="transparent" />
                <circle
                  cx={cx}
                  cy={cy}
                  r={10}
                  fill="#fff"
                  stroke={color}
                  strokeWidth={2}
                  strokeDasharray={c.pose ? undefined : "3 2.5"}
                  filter="drop-shadow(0 1px 2px rgb(16 24 40 / 0.18))"
                />
                <circle cx={cx + fx * 4} cy={cy + fy * 4} r={3} fill={color} />
              </g>
            </g>
          );
        })}
      </svg>

      {/* name labels */}
      {layoutLabels(cameras, poses, sx, sy).map(({ c, x, y, below }) => {
        return (
          <div
            key={c.id}
            className="pointer-events-none absolute left-0 top-0"
            style={{ transform: `translate(${x}px, ${y}px) translate(-50%, ${below ? "0" : "-100%"})` }}
          >
            <div
              className={`whitespace-nowrap rounded-md bg-surface px-2 py-0.5 text-[11px] font-medium shadow-[var(--shadow-float)] ${
                c.id === selected ? "text-accent" : "text-fg"
              }`}
            >
              {c.label}
              {!c.pose && <span className="ml-1 font-normal text-faint">· not placed</span>}
            </div>
          </div>
        );
      })}

      {/* north + scale */}
      <div className="pointer-events-none absolute left-4 top-4 flex flex-col items-center rounded-md bg-surface/90 px-2 py-1.5 text-[10px] font-semibold text-muted shadow-[var(--shadow-float)]">
        <svg width="10" height="12" viewBox="0 0 10 12">
          <path d="M5 0L10 12L5 9L0 12Z" fill="currentColor" />
        </svg>
        N
      </div>
      <div className="pointer-events-none absolute left-[60px] top-4 flex flex-col items-start gap-1 text-[11px] text-muted">
        <span>{scaleBar.m} m</span>
        <div className="h-1.5 border-x border-b border-current" style={{ width: scaleBar.px }} />
      </div>
      <button
        onClick={fit}
        title="Fit all cameras"
        className="absolute left-4 top-[68px] flex h-8 w-8 items-center justify-center rounded-md bg-surface text-muted shadow-[var(--shadow-float)] hover:text-fg"
      >
        <Maximize2 size={14} />
      </button>

      {/* site tools: map photo + house outlines */}
      {onSiteSave && (
        <div className="absolute bottom-4 right-4 flex items-center gap-1 rounded-lg bg-surface p-1 text-[12px] shadow-[var(--shadow-float)]">
          <input
            ref={fileInput}
            type="file"
            accept="image/png,image/jpeg"
            className="hidden"
            onChange={async (e) => {
              const f = e.target.files?.[0];
              e.target.value = "";
              if (!f || !onUploadMap) return;
              try {
                await onUploadMap(f);
                setMode("align");
                setNotice(null);
                void runAutoAlign();
              } catch (err) {
                setNotice(String((err as Error).message ?? err));
              }
            }}
          />
          {!site?.map ? (
            <ToolButton icon={<ImagePlus size={14} />} label="Map photo" onClick={() => fileInput.current?.click()} />
          ) : (
            <>
              <ToolButton icon={<Move size={14} />} label="Align photo" active={mode === "align"} onClick={() => switchMode("align")} />
              {mode === "align" && (
                <ToolButton icon={<Sparkles size={14} />} label={aligning ? "Aligning…" : "Auto-align"} onClick={() => !aligning && runAutoAlign()} />
              )}
              {mode === "align" && undoMap && !aligning && (
                <ToolButton
                  icon={<Undo2 size={14} />}
                  label="Undo"
                  onClick={() => {
                    onSiteSave({ ...site, map: undoMap });
                    setUndoMap(null);
                    setNotice("Previous placement restored");
                  }}
                />
              )}
              <ToolButton icon={<MousePointerClick size={14} />} label="Click a roof" active={mode === "roof"} onClick={() => switchMode("roof")} />
              <ToolButton icon={<Hexagon size={14} />} label="Trace house" active={mode === "trace"} onClick={() => switchMode("trace")} />
              {(mode === "trace" || mode === "roof") && !!site.houses.length && (
                <>
                  <ToolButton
                    icon={<Undo2 size={14} />}
                    label="Undo house"
                    onClick={() => {
                      onSiteSave({ ...site, houses: site.houses.slice(0, -1) });
                      setSelHouse(null);
                    }}
                  />
                  <ToolButton
                    icon={<Trash2 size={14} />}
                    label={confirmClear ? `Remove all ${site.houses.length}?` : "Clear houses"}
                    active={confirmClear}
                    onClick={() => {
                      if (!confirmClear) return setConfirmClear(true);
                      onSiteSave({ ...site, houses: [] });
                      setSelHouse(null);
                      setConfirmClear(false);
                      setNotice("All houses removed");
                    }}
                  />
                </>
              )}
              {mode === "align" && (
                <>
                  <label className="flex h-8 items-center gap-2 px-2 text-muted" title="Photo opacity">
                    Photo
                    <input
                      type="range"
                      min={0.15}
                      max={1}
                      step={0.05}
                      value={photoOpacity}
                      onChange={(e) => setPhotoOpacity(Number(e.target.value))}
                      className="w-20 accent-[#006fff]"
                    />
                  </label>
                  <ToolButton icon={<Trash2 size={14} />} label="Remove photo" onClick={() => onRemoveMap?.()} />
                </>
              )}
            </>
          )}
        </div>
      )}
      {(mode !== "none" || notice) && (
        <div className="pointer-events-none absolute bottom-[60px] right-4 max-w-[60%] rounded-md bg-surface px-3 py-1.5 text-[12px] text-fg shadow-[var(--shadow-float)]">
          {notice && (mode === "none" || mode === "roof" || aligning || (mode === "align" && pairs.length === 0))
            ? notice
            : mode === "roof"
              ? "Click on a roof in the photo to outline it"
            : mode === "align"
              ? pairs.length === 0 || pairs[pairs.length - 1].world
                ? `${notice ? notice + " · " : ""}${
                    pairs.length
                      ? "Click the second spot on the photo"
                      : "Drag to move · corners to resize · round handle to rotate · keys: arrows, + −, Q E · or click two matching spots"
                  }`
                : "Now click where that spot really is on the plan (cameras snap)"
              : trace.length === 0
                ? `${notice ? notice + " · " : ""}Click the corners of a roof · Enter or click the first corner to finish`
                : `${trace.length} corner${trace.length > 1 ? "s" : ""} · Enter to finish · Backspace to undo · Esc to cancel`}
        </div>
      )}
    </div>
  );
}

function ToolButton({ icon, label, active, onClick }: { icon: React.ReactNode; label: string; active?: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={`flex h-8 items-center gap-1.5 rounded-md px-2.5 font-medium ${active ? "bg-accent text-white" : "text-muted hover:bg-black/5 hover:text-fg"}`}
    >
      {icon}
      {label}
    </button>
  );
}

const LABEL_H = 22;

/**
 * Label positions: on the side of each camera away from where it looks, nudged apart vertically
 * so cameras mounted close together don't hide each other's names.
 */
function layoutLabels(
  cameras: Camera[],
  poses: Record<string, Pose>,
  sx: (x: number) => number,
  sy: (z: number) => number,
) {
  type Box = { c: Camera; x: number; y: number; below: boolean; w: number };
  const boxes: Box[] = [];
  for (const c of cameras) {
    const p = poses[c.id];
    if (!p) continue;
    const below = Math.cos(rad(p.yaw)) < 0.3;
    const w = c.label.length * 6.6 + (c.pose ? 18 : 80);
    boxes.push({ c, x: sx(p.x), y: sy(p.z) + (below ? 18 : -18), below, w });
  }
  // Box top for overlap tests (labels hang below / sit above their anchor).
  const top = (b: Box) => (b.below ? b.y : b.y - LABEL_H);
  const placed: Box[] = [];
  for (const b of [...boxes].sort((a, b) => top(a) - top(b))) {
    for (let i = 0; i < 12; i++) {
      const hit = placed.some((o) => Math.abs(o.x - b.x) < (o.w + b.w) / 2 && Math.abs(top(o) - top(b)) < LABEL_H + 2);
      if (!hit) break;
      b.y += b.below ? LABEL_H + 2 : -(LABEL_H + 2);
    }
    placed.push(b);
  }
  return boxes;
}
