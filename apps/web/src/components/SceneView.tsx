import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Box, Loader2, Map as MapIcon, RotateCw, Undo2 } from "lucide-react";
import { api, bestProfile, useSource, type Camera, type LiveSnapshot, type Pose, type useLive } from "../api";
import { LiveImage } from "./LiveImage";
import { heading, PlanView } from "./PlanView";
import { normYaw, placeholderPose, Scene } from "./Scene";
import { useGroundMaps, useSite, useWorld } from "../sceneData";
import { SelectedCard } from "./Views";

type Mode = "3d" | "plan";

/**
 * The scene, shown two ways: the 3D view (look around) and the floor plan (arrange cameras by
 * dragging). Poses live in `drafts` for instant feedback and are saved to the server, debounced.
 */
export function SceneView({
  cameras,
  live,
  selected,
  onSelect,
  onChanged,
  mode,
  onModeChange,
}: {
  cameras: Camera[];
  /** The live engine's feed (shared with the rest of the app). */
  live: ReturnType<typeof useLive>;
  selected: string | null;
  onSelect: (id: string | null) => void;
  onChanged: () => void;
  /** 3D or plan — owned by the address bar (/3d, /plan). */
  mode: Mode;
  onModeChange: (m: Mode) => void;
}) {
  const modeRef = useRef(mode);
  modeRef.current = mode;
  const setMode = (m: Mode | ((cur: Mode) => Mode)) => onModeChange(typeof m === "function" ? m(modeRef.current) : m);
  const [drafts, setDrafts] = useState<Record<string, Pose>>({});
  const scenes = useGroundMaps(cameras);
  const siteState = useSite();
  const site = siteState.site;
  // The scene drops camera-guessed walls inside traced houses, so refetch when houses change.
  const world = useWorld(cameras, JSON.stringify(site?.houses ?? []));
  const timers = useRef<Record<string, ReturnType<typeof setTimeout>>>({});

  const poses = useMemo(() => {
    const out: Record<string, Pose> = {};
    cameras.forEach((c, i) => (out[c.id] = drafts[c.id] ?? c.pose ?? placeholderPose(i, cameras.length, c)));
    return out;
  }, [cameras, drafts]);

  const unplaced = cameras.filter((c) => !c.pose && !drafts[c.id]).length;
  const selectedCam = cameras.find((c) => c.id === selected);

  const save = useCallback(
    (id: string, pose: Pose | null, delay = 350) => {
      clearTimeout(timers.current[id]);
      timers.current[id] = setTimeout(async () => {
        await api.update(id, { pose });
        onChanged();
      }, delay);
    },
    [onChanged],
  );

  const setPose = useCallback(
    (id: string, pose: Pose, persist = true) => {
      setDrafts((d) => ({ ...d, [id]: pose }));
      if (persist) save(id, pose);
    },
    [save],
  );
  const commitPose = useCallback((id: string, p: Pose) => setPose(id, p), [setPose]);
  const previewPose = useCallback((id: string, p: Pose) => setPose(id, p, false), [setPose]);

  function resetPlacement(id: string) {
    setDrafts((d) => {
      const { [id]: _, ...rest } = d;
      return rest;
    });
    save(id, null, 0);
  }

  // P toggles plan / 3D; Esc clears the selection.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.tagName === "INPUT") return;
      if (e.key === "p" && !e.metaKey && !e.ctrlKey) setMode((m) => (m === "plan" ? "3d" : "plan"));
      if (e.key === "Escape") onSelect(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onSelect]);

  const tab = (m: Mode, icon: React.ReactNode, label: string, hint: string) => (
    <button
      onClick={() => setMode(m)}
      title={hint}
      className={`flex h-8 items-center gap-1.5 rounded-md px-3 text-[13px] transition ${
        mode === m ? "bg-accent text-white" : "text-fg hover:bg-bg"
      }`}
    >
      {icon} {label}
    </button>
  );

  return (
    <div className="relative min-w-0 flex-1 overflow-hidden rounded-[10px] border border-line">
      {mode === "3d" ? (
        <Scene cameras={cameras} poses={poses} selected={selected} onSelect={onSelect} live={live.ref} scenes={scenes} world={world} houses={site?.houses} />
      ) : (
        <PlanView
          cameras={cameras}
          poses={poses}
          selected={selected}
          onSelect={onSelect}
          onPoseChange={previewPose}
          onPoseCommit={commitPose}
          objects={live.snapshot?.objects}
          scenes={scenes}
          world={world}
          site={site}
          siteImg={siteState.img}
          onSiteSave={siteState.save}
          onUploadMap={siteState.upload}
          onRemoveMap={siteState.removeMap}
          onAutoAlign={siteState.autoAlign}
          onFindRoof={siteState.findRoof}
        />
      )}

      <div className="absolute right-4 top-4 flex items-center gap-1 rounded-[10px] bg-surface p-1 shadow-[var(--shadow-float)]">
        {tab("3d", <Box size={14} />, "3D", "Look around the scene (P)")}
        {tab("plan", <MapIcon size={14} />, "Plan", "Arrange cameras on the floor plan (P)")}
      </div>

      <LiveStatus snapshot={live.snapshot} unplaced={cameras.filter((c) => !c.pose).length} />

      {mode === "3d" && (
        <div className="absolute bottom-4 left-4 flex flex-col items-start gap-2">
          {!selectedCam && unplaced > 0 && (
            <button
              onClick={() => setMode("plan")}
              className="rounded-md bg-surface/90 px-3 py-2 text-left text-[12px] text-muted shadow-[var(--shadow-float)] hover:text-fg"
            >
              Placeholder layout · <span className="font-medium text-accent">Arrange cameras in Plan</span>
            </button>
          )}
        </div>
      )}

      {mode === "plan" && (
        <div className="pointer-events-none absolute bottom-4 left-4 max-w-[calc(100%-340px)] rounded-md bg-surface/90 px-3 py-2 text-[12px] text-muted shadow-[var(--shadow-float)]">
          Drag a camera to move it · drag its view to turn it · scroll to zoom
        </div>
      )}

      {mode === "plan" && selectedCam && (
        <PoseCard
          camera={selectedCam}
          cameras={cameras}
          pose={poses[selectedCam.id]}
          onChange={(p) => setPose(selectedCam.id, p)}
          onReset={() => resetPlacement(selectedCam.id)}
        />
      )}
      {mode === "3d" && selectedCam && <SelectedCard camera={selectedCam} onClose={() => onSelect(null)} />}
    </div>
  );
}

const PLURAL: Record<string, [string, string]> = {
  person: ["person", "people"],
  vehicle: ["vehicle", "vehicles"],
  bike: ["bike", "bikes"],
  animal: ["animal", "animals"],
};
const GROUP: Record<string, string> = {
  person: "person",
  car: "vehicle",
  truck: "vehicle",
  bus: "vehicle",
  bicycle: "bike",
  motorcycle: "bike",
  cat: "animal",
  dog: "animal",
};

/** What the server's live engine is doing — counts when running, progress or errors otherwise. */
function LiveStatus({ snapshot, unplaced }: { snapshot: LiveSnapshot | null; unplaced: number }) {
  if (!snapshot || snapshot.state === "off") return null;
  const counts = new Map<string, number>();
  for (const o of snapshot.objects) {
    if (o.class === "bin") continue; // part of the scene, not something moving
    const g = GROUP[o.class] ?? "animal";
    counts.set(g, (counts.get(g) ?? 0) + 1);
  }
  const failing = Object.values(snapshot.cameras).filter((c) => c.error && c.fps === 0).length;
  const busy = snapshot.state === "starting" || snapshot.state === "downloading" || snapshot.state === "loading";
  const source = useSource().data;
  const replay = source?.replay ? (source.replays.find((r) => r.name === source.replay)?.title ?? source.replay) : null;
  return (
    <div className="pointer-events-none absolute left-4 top-4 flex max-w-[calc(100%-200px)] items-center gap-2 rounded-md bg-surface/95 px-3 py-2 text-[12px] shadow-[var(--shadow-float)]">
      {busy ? (
        <>
          <Loader2 size={13} className="animate-spin text-muted" />
          <span className="text-muted">{snapshot.message || "Starting live view"}</span>
        </>
      ) : snapshot.state === "error" ? (
        <span className="text-bad">Live view stopped: {snapshot.message}</span>
      ) : (
        <>
          <span className="relative flex h-2 w-2">
            <span className={`absolute inline-flex h-full w-full animate-ping rounded-full opacity-60 ${replay ? "bg-warn" : "bg-ok"}`} />
            <span className={`relative inline-flex h-2 w-2 rounded-full ${replay ? "bg-warn" : "bg-ok"}`} />
          </span>
          <span className="font-medium">{replay ? `Test video: ${replay}` : "Live"}</span>
          <span className="text-muted">
            {counts.size === 0
              ? "· nothing moving"
              : [...counts].map(([g, n]) => `· ${n} ${PLURAL[g][n === 1 ? 0 : 1]}`).join(" ")}
          </span>
          <span className="text-faint">
            · {Object.keys(snapshot.cameras).length} cams
            {failing > 0 && <span className="text-warn"> ({failing} offline)</span>}
            {unplaced > 0 && ` · ${unplaced} not placed`}
            {snapshot.infer_ms > 0 && ` · ${snapshot.accelerated ? "GPU" : "CPU"} ${Math.round(snapshot.infer_ms)} ms`}
          </span>
        </>
      )}
    </div>
  );
}

const COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];

/** What the plan can't show by dragging: tilt and mounting height — next to the live view. */
function PoseCard({
  camera,
  cameras,
  pose,
  onChange,
  onReset,
}: {
  camera: Camera;
  cameras: Camera[];
  pose: Pose;
  onChange: (p: Pose) => void;
  onReset: () => void;
}) {
  const p = bestProfile(camera);
  const h = heading(pose.yaw);
  const cal = camera.calibration;
  const [showCal, setShowCal] = useState(false);
  const aspect = p?.width && p?.height ? `${p.width} / ${p.height}` : "16 / 9";
  const slider = (
    label: string,
    value: number,
    min: number,
    max: number,
    step: number,
    fmt: (v: number) => string,
    set: (v: number) => void,
  ) => (
    <label className="block py-1.5">
      <div className="mb-1 flex items-baseline justify-between text-[13px]">
        <span className="text-muted">{label}</span>
        <span className="tabular-nums">{fmt(value)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => set(Number(e.target.value))}
        className="w-full accent-accent"
      />
    </label>
  );

  return (
    <div className="absolute bottom-4 right-4 w-[min(300px,calc(100%-32px))] overflow-hidden rounded-[10px] bg-surface shadow-[var(--shadow-float)]">
      <div className="relative">
        {showCal && cal ? (
          <img
            src={api.calibrationUrl(camera.id, cal.computed_at)}
            alt="Calibration check"
            className="w-full object-cover"
            style={{ aspectRatio: aspect }}
          />
        ) : (
          <LiveImage src={api.snapshotUrl(camera.id)} light className="w-full" style={{ aspectRatio: aspect }} />
        )}
        {cal && (
          <div className="absolute right-2 top-2 flex rounded-md bg-black/45 p-0.5 text-[11px] text-white backdrop-blur">
            {(["Live", "Calibration"] as const).map((t) => (
              <button
                key={t}
                onClick={() => setShowCal(t === "Calibration")}
                className={`rounded px-1.5 py-0.5 ${showCal === (t === "Calibration") ? "bg-white/25" : "opacity-75 hover:opacity-100"}`}
              >
                {t}
              </button>
            ))}
          </div>
        )}
      </div>
      <div className="px-4 pb-3 pt-3">
        <div className="mb-1 flex items-center justify-between gap-2">
          <span className="truncate text-[14px] font-medium">{camera.label}</span>
          <button
            onClick={onReset}
            title="Back to the placeholder ring"
            className="flex shrink-0 items-center gap-1 text-[12px] text-muted hover:text-fg"
          >
            <Undo2 size={12} /> Reset
          </button>
        </div>

        <div className="flex items-center justify-between py-1.5 text-[13px]">
          <span className="text-muted">Facing</span>
          <span className="flex items-center gap-2">
            <span className="tabular-nums">
              {COMPASS[Math.round(h / 45) % 8]} · {h}°
            </span>
            <button
              onClick={() => onChange({ ...pose, yaw: normYaw(pose.yaw + 180) })}
              title="Turn 180°"
              className="flex h-6 items-center gap-1 rounded border border-line-strong px-1.5 text-[12px] transition hover:bg-bg"
            >
              <RotateCw size={11} /> 180°
            </button>
          </span>
        </div>
        {cal && (
          <div className="mb-1 rounded-md bg-sunken px-2.5 py-2 text-[12px] leading-relaxed text-muted">
            <span className="font-medium text-fg">Measured from the image</span> · tilt {cal.pitch.toFixed(1)}° · lean{" "}
            {cal.roll.toFixed(1)}° · height {cal.height.toFixed(2)} m ·{" "}
            {cal.lenses.length > 1
              ? `${cal.lenses.length} lenses × ${Math.round(cal.lenses[0].hfov)}°`
              : `${Math.round(cal.lenses[0].hfov)}° lens`}
            <div className="mt-1">
              {cal.alignment?.anchor
                ? "Reference camera: the others are aligned to it."
                : cal.alignment
                  ? `Placed from ${cal.alignment.inliers} points matched with ${
                      cameras.find((c) => c.id === cal.alignment?.reference)?.label ?? "another camera"
                    } (±${cal.alignment.rms.toFixed(2)} m).`
                  : "Heading and position as you placed them — not enough overlap with another camera to align."}
            </div>
            {showCal && (
              <div className="mt-1 text-faint">
                Colours: road, footpath, grass, buildings, trees, vehicles. Yellow = computed horizon; cyan = true
                verticals — they should run parallel to walls and poles.
              </div>
            )}
          </div>
        )}
        {slider("Tilt down", pose.pitch, 0, 90, 1, (v) => `${v}°`, (pitch) => onChange({ ...pose, pitch }))}
        {slider("Mounting height", pose.height, 0.5, 8, 0.1, (v) => `${v.toFixed(1)} m`, (height) =>
          onChange({ ...pose, height }),
        )}
        <div className="mt-1 border-t border-line pt-2 text-[11px] leading-relaxed text-faint">
          Arrows nudge · Q/E turn · Shift for bigger steps. Rough is fine; the view's reach on the plan shows the tilt.
        </div>
      </div>
    </div>
  );
}
