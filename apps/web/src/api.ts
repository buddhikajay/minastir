import { useCallback, useEffect, useRef, useState } from "react";

export type Profile = {
  token: string;
  name: string;
  encoding: string | null;
  width: number | null;
  height: number | null;
  stream_uri: string | null;
  snapshot_uri: string | null;
};

/** Placement in metres; yaw 0 looks along +Z (counter-clockwise from above), pitch = downward tilt. */
export type Pose = { x: number; z: number; height: number; yaw: number; pitch: number };

export type Camera = {
  id: string;
  /** "onvif" = adopted over ONVIF; "stream" = added by RTSP/RTSPS URL (device_url is the stream). */
  kind: "onvif" | "stream";
  /** Display name: user's name → name found on camera/network → model. */
  label: string;
  model_label: string;
  /** Best non-default name the camera or network reports. */
  detected_name: string | null;
  /** Name set in Minastir. */
  name: string | null;
  /** Hand-placed position in the scene; null = not placed (shown on a placeholder ring). */
  pose: Pose | null;
  /** Lens, tilt, lean and height measured from the camera's own image. */
  calibration: Calibration | null;
  has_background: boolean;
  ip: string;
  device_url: string;
  manufacturer: string | null;
  model: string | null;
  firmware: string | null;
  serial: string | null;
  ptz: boolean;
  profiles: Profile[];
};

export type LensCalibration = {
  u0: number;
  u1: number;
  hfov: number;
  vfov: number;
  yaw_offset: number;
  height: number;
  ground_slope: number;
};
export type Alignment = {
  anchor: boolean;
  reference: string | null;
  inliers: number;
  rms: number;
  /** The hand placement before alignment: [yaw, x, z]. */
  rough: [number, number, number];
};
export type Calibration = {
  lenses: LensCalibration[];
  pitch: number;
  roll: number;
  height: number;
  computed_at: number;
  /** Depth-scale correction from aligning with other cameras. */
  scale: number;
  alignment: Alignment | null;
};

export type Job = { running: boolean; message: string; done: number; total: number };
export type Status = { cameras: number; captured: number; job: Job };
export type Discovered = {
  endpoint: string;
  ip: string;
  device_url: string;
  name: string | null;
  hardware: string | null;
  /** Non-default name from discovery or reverse DNS. */
  display_name: string | null;
  adopted_id: string | null;
};

export type AdoptTarget = { device_url: string } | { host: string };

export const SERVER_UNREACHABLE = "Can't reach the Minastir server";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`/api${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
    });
  } catch {
    throw new Error(SERVER_UNREACHABLE);
  }
  // The Vite dev proxy answers 5xx with an empty/HTML body when the Rust server is down.
  if (res.status >= 502 && res.status <= 504 && !res.headers.get("content-type")?.includes("json")) {
    throw new Error(SERVER_UNREACHABLE);
  }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error ?? `${res.status} ${res.statusText}`);
  return body as T;
}

/** A test video: a folder of recorded camera files the engine can play instead of the cameras. */
export type Replay = { name: string; cameras: string[]; title: string | null };
/** What the live engine is watching: the cameras (replay null) or a test video. */
export type Source = { replay: string | null; replays: Replay[] };

export const api = {
  source: () => request<Source>("/source"),
  setSource: (replay: string | null) => request<Source>("/source", { method: "PUT", body: JSON.stringify({ replay }) }),
  liveFrameUrl: (id: string, width?: number) => `/api/live/${id}/frame.jpg${width ? `?width=${width}` : ""}`,
  status: () => request<Status>("/status"),
  cameras: () => request<Camera[]>("/cameras"),
  discover: () => request<Discovered[]>("/discover"),
  adopt: (target: AdoptTarget, user: string, pass: string) =>
    request<Camera>("/adopt", { method: "POST", body: JSON.stringify({ ...target, user, pass }) }),
  addStream: (url: string, name: string) =>
    request<Camera>("/streams", { method: "POST", body: JSON.stringify({ url, name }) }),
  update: (id: string, patch: { name?: string | null; pose?: Pose | null }) =>
    request<Camera>(`/cameras/${id}`, { method: "PATCH", body: JSON.stringify(patch) }),
  rename: (id: string, name: string | null) =>
    request<Camera>(`/cameras/${id}`, { method: "PATCH", body: JSON.stringify({ name }) }),
  refresh: (id: string) => request<Camera>(`/cameras/${id}/refresh`, { method: "POST" }),
  remove: (id: string) => request<unknown>(`/cameras/${id}`, { method: "DELETE" }),
  capture: () => request<Job>("/capture", { method: "POST" }),
  calibrate: () => request<Job>("/calibrate", { method: "POST" }),
  calibrationUrl: (id: string, at: number) => `/api/cameras/${id}/calibration.jpg?v=${at}`,
  snapshotUrl: (id: string) => `/api/cameras/${id}/snapshot`,
  backgroundUrl: (id: string) => `/api/cameras/${id}/background`,
};

/** Fetch on mount and every `ms` (if given); exposes reload(). */
export function usePoll<T>(fn: () => Promise<T>, ms?: number) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fnRef = useRef(fn);
  fnRef.current = fn;

  const reload = useCallback(async () => {
    try {
      setData(await fnRef.current());
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    reload();
    if (!ms) return;
    const t = setInterval(reload, ms);
    return () => clearInterval(t);
  }, [reload, ms]);

  return { data, error, reload };
}

export function bestProfile(c: Camera): Profile | undefined {
  return [...c.profiles]
    .filter((p) => p.stream_uri)
    .sort((a, b) => (b.width ?? 0) * (b.height ?? 0) - (a.width ?? 0) * (a.height ?? 0))[0];
}

export function resolution(c: Camera): string {
  const p = bestProfile(c);
  return p?.width && p?.height ? `${p.width}×${p.height}` : "—";
}

// ---------- live objects (computed by the server; the UI only renders) ----------

export type LiveObject = {
  id: number;
  class: "person" | "bicycle" | "car" | "motorcycle" | "bus" | "truck" | "cat" | "dog" | "bin";
  x: number;
  z: number;
  /** Direction of travel, same convention as camera yaw (degrees). */
  heading: number;
  /** Whether heading is known: measured from motion, or fitted while parked. */
  heading_known?: boolean;
  /** Whether the nose is known (a fitted parked car may only know its axis). */
  facing_known?: boolean;
  /** Render-and-compare match (0–1) while parked, when fitted. */
  fit?: number | null;
  /** Appearance fingerprints gathered so far. */
  looks?: number;
  /** The describer's probabilities that a detected vehicle is a car, truck, bin or other. */
  verdict?: [number, number, number, number] | null;
  speed: number;
  score: number;
  cameras: string[];
};

export type LiveDetection = { class: string; score: number; bbox: [number, number, number, number] };

export type LiveSnapshot = {
  time: number;
  state: "off" | "starting" | "downloading" | "loading" | "running" | "error";
  message: string;
  accelerated: boolean;
  infer_ms: number;
  objects: LiveObject[];
  cameras: Record<string, { fps: number; error: string | null; detections: LiveDetection[] }>;
};

/**
 * Subscribes to the server's live snapshots. `ref` always holds the newest one (read it in render
 * loops); `snapshot` is React state for UI that should re-render on updates.
 */
export function useLive() {
  const ref = useRef<LiveSnapshot | null>(null);
  const [snapshot, setSnapshot] = useState<LiveSnapshot | null>(null);
  useEffect(() => {
    let ws: WebSocket | null = null;
    let retry: ReturnType<typeof setTimeout>;
    let alive = true;
    const connect = () => {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      ws = new WebSocket(`${proto}://${location.host}/api/live`);
      ws.onmessage = (e) => {
        try {
          const s = JSON.parse(e.data) as LiveSnapshot;
          ref.current = s;
          setSnapshot(s);
        } catch {
          /* ignore malformed frames */
        }
      };
      ws.onclose = () => {
        if (alive) retry = setTimeout(connect, 2000);
      };
    };
    connect();
    return () => {
      alive = false;
      clearTimeout(retry);
      ws?.close();
    };
  }, []);
  return { ref, snapshot };
}

// ---------- what the live engine watches (shared by every component that shows it) ----------

type SourceState = { data: Source | null; busy: boolean; error: string | null };
let sourceState: SourceState = { data: null, busy: false, error: null };
const sourceListeners = new Set<() => void>();
const setSourceState = (s: Partial<SourceState>) => {
  sourceState = { ...sourceState, ...s };
  sourceListeners.forEach((l) => l());
};
let sourceLoaded = false;

async function pickSource(replay: string | null) {
  setSourceState({ busy: true, error: null });
  try {
    setSourceState({ data: await api.setSource(replay), busy: false });
  } catch (e) {
    setSourceState({ busy: false, error: e instanceof Error ? e.message : String(e) });
  }
}

/** The cameras or a test video, and a way to switch: one shared state for the whole app. */
export function useSource() {
  const [state, setState] = useState(sourceState);
  useEffect(() => {
    const l = () => setState(sourceState);
    sourceListeners.add(l);
    if (!sourceLoaded) {
      sourceLoaded = true;
      api.source().then(
        (data) => setSourceState({ data }),
        () => (sourceLoaded = false),
      );
    }
    l();
    return () => void sourceListeners.delete(l);
  }, []);
  return { ...state, pick: pickSource };
}
