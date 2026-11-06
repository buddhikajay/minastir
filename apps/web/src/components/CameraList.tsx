import { useEffect, useRef, useState } from "react";
import { Film, Radio } from "lucide-react";
import { api, bestProfile, useSource, type Camera, type LiveSnapshot, type Source } from "../api";
import { LiveImage } from "./LiveImage";

/**
 * The cameras, as the live engine sees them: each row's thumbnail is the frame being analysed
 * (the test video's, when one is playing); hover a row for a large live view with the current
 * detections drawn on. The header switches between the cameras and a test video.
 */
export function CameraList({
  cameras,
  live,
  selected,
  onSelect,
}: {
  cameras: Camera[];
  live: LiveSnapshot | null;
  selected: string | null;
  onSelect: (id: string | null) => void;
}) {
  const source = useSource();
  const [peek, setPeek] = useState<{ id: string; rect: DOMRect } | null>(null);
  const hoverTimer = useRef<ReturnType<typeof setTimeout>>(undefined);
  useEffect(() => () => clearTimeout(hoverTimer.current), []);

  const analysed = (id: string) => (live?.cameras[id]?.fps ?? 0) > 0;
  const replay = source.data?.replay ?? null;
  const inReplay = (id: string) => !replay || !!source.data?.replays.find((r) => r.name === replay)?.cameras.includes(id);

  const enter = (id: string, el: HTMLElement) => {
    clearTimeout(hoverTimer.current);
    // A short delay, so sweeping the pointer down the list doesn't flash every camera.
    const rect = el.getBoundingClientRect();
    hoverTimer.current = setTimeout(() => setPeek({ id, rect }), peek ? 0 : 180);
  };
  const leave = () => {
    clearTimeout(hoverTimer.current);
    setPeek(null);
  };

  const peekCam = peek && cameras.find((c) => c.id === peek.id);

  return (
    <section className="rounded-[10px] border border-line bg-surface py-2">
      <div className="flex items-center gap-2 px-5 pb-1.5 pt-2">
        <span className="text-[14px] font-medium">Cameras</span>
        <span className="text-[12px] text-muted">{cameras.length}</span>
        <SourcePicker source={source.data} busy={source.busy} error={source.error} onPick={source.pick} />
      </div>
      {cameras.map((c) => {
        const running = analysed(c.id);
        const absent = !inReplay(c.id);
        const err = live?.cameras[c.id]?.error;
        return (
          <button
            key={c.id}
            onClick={() => onSelect(selected === c.id ? null : c.id)}
            onMouseEnter={(e) => enter(c.id, e.currentTarget)}
            onMouseLeave={leave}
            className={`flex w-full items-center gap-3 px-5 py-2 text-left transition ${
              selected === c.id ? "bg-accent/[0.06]" : "hover:bg-bg"
            }`}
          >
            <div className={`relative h-[34px] w-[60px] shrink-0 overflow-hidden rounded-[4px] ${absent ? "opacity-40" : ""}`}>
              <LiveImage
                key={running ? "live" : "snap"}
                src={running ? api.liveFrameUrl(c.id, 240) : api.snapshotUrl(c.id)}
                interval={running ? 1000 : 5000}
                light
                className="h-full w-full"
              />
            </div>
            <div className="min-w-0 flex-1">
              <div className={`truncate text-[13px] font-medium ${selected === c.id ? "text-accent" : ""} ${absent ? "text-muted" : ""}`}>
                {c.label}
              </div>
              <div className="truncate text-[11px] text-faint">
                {absent ? "Not in this test video" : err ? <span className="text-bad">{err}</span> : <span className="font-mono">{c.ip}</span>}
              </div>
            </div>
            <span
              title={running ? "Being analysed" : absent ? "Not in this test video" : err ? "Error" : "Not analysed"}
              className={`h-1.5 w-1.5 shrink-0 rounded-full ${running ? "bg-ok" : err ? "bg-bad" : "bg-faint"}`}
            />
          </button>
        );
      })}
      {peek && peekCam && <Peek camera={peekCam} rect={peek.rect} live={live} analysed={analysed(peekCam.id)} replay={inReplay(peekCam.id) ? replay : null} replayTitle={source.data?.replays.find((r) => r.name === replay)?.title ?? null} />}
    </section>
  );
}

function SourcePicker({
  source,
  busy,
  error,
  onPick,
}: {
  source: Source | null;
  busy: boolean;
  error: string | null;
  onPick: (replay: string | null) => void;
}) {
  if (!source) return null;
  const testing = source.replay !== null;
  const Icon = testing ? Film : Radio;
  return (
    <label
      title={error ?? (testing ? "Playing a test video instead of the cameras" : "Watching the cameras live")}
      className={`relative ml-auto flex h-7 items-center gap-1.5 rounded-md border pl-2 pr-1 text-[12px] transition ${
        error ? "border-bad text-bad" : testing ? "border-warn/50 bg-warn/[0.08] text-fg" : "border-line-strong text-fg"
      } ${busy ? "opacity-60" : "hover:bg-bg"}`}
    >
      <Icon size={13} className={testing ? "text-warn" : "text-ok"} />
      <select
        value={source.replay ?? ""}
        disabled={busy}
        onChange={(e) => onPick(e.target.value || null)}
        aria-label="What the live engine watches"
        className="max-w-[150px] cursor-pointer appearance-none truncate bg-transparent pr-4 outline-none"
      >
        <option value="">Live cameras</option>
        <optgroup label="Test videos">
          {source.replays.map((r) => (
            <option key={r.name} value={r.name}>
              {r.title ?? r.name}
            </option>
          ))}
        </optgroup>
      </select>
      <svg className="pointer-events-none absolute right-1.5 text-muted" width="10" height="10" viewBox="0 0 10 10" aria-hidden>
        <path d="M2 3.5 5 6.5 8 3.5" fill="none" stroke="currentColor" strokeWidth="1.4" />
      </svg>
    </label>
  );
}

const PEEK_W = 560;

/** The enlarged live view beside the list, with the engine's detections drawn on. */
function Peek({
  camera,
  rect,
  live,
  analysed,
  replay,
  replayTitle,
}: {
  camera: Camera;
  rect: DOMRect;
  live: LiveSnapshot | null;
  analysed: boolean;
  replay: string | null;
  replayTitle: string | null;
}) {
  const p = bestProfile(camera);
  const aspect = p?.width && p?.height ? p.width / p.height : 16 / 9;
  const width = Math.min(PEEK_W, window.innerWidth - rect.right - 32);
  const height = width / aspect + 40;
  const top = Math.max(12, Math.min(rect.top + rect.height / 2 - height / 2, window.innerHeight - height - 12));
  const dets = analysed ? (live?.cameras[camera.id]?.detections ?? []) : [];
  const people = dets.filter((d) => d.class === "person").length;
  const others = dets.length - people;
  if (width < 200) return null;
  return (
    <div
      className="pointer-events-none fixed z-50 overflow-hidden rounded-[10px] bg-surface shadow-[var(--shadow-float)]"
      style={{ left: rect.right + 12, top, width, animation: "drawer-in 120ms ease-out" }}
    >
      <div className="relative bg-video" style={{ aspectRatio: aspect }}>
        <LiveImage
          key={analysed ? "live" : "snap"}
          src={analysed ? api.liveFrameUrl(camera.id, 960) : api.snapshotUrl(camera.id)}
          interval={analysed ? 120 : 1500}
          className="absolute inset-0 h-full w-full"
        />
        {dets.map((d, k) => {
          const [x0, y0, x1, y1] = d.bbox;
          return (
            <div
              key={k}
              className={`absolute rounded-[2px] border ${d.class === "person" ? "border-accent" : "border-white/80"}`}
              style={{ left: `${x0 * 100}%`, top: `${y0 * 100}%`, width: `${(x1 - x0) * 100}%`, height: `${(y1 - y0) * 100}%` }}
            />
          );
        })}
        {replay && (
          <span className="absolute left-2 top-2 flex items-center gap-1 rounded bg-black/55 px-1.5 py-0.5 text-[11px] text-white">
            <Film size={11} /> {replayTitle ?? replay}
          </span>
        )}
      </div>
      <div className="flex h-10 items-center gap-3 px-3.5 text-[12px]">
        <span className="truncate text-[13px] font-medium">{camera.label}</span>
        <span className="ml-auto shrink-0 text-muted">
          {analysed
            ? `${people} ${people === 1 ? "person" : "people"}${others ? ` · ${others} other` : ""} · ${(live?.cameras[camera.id]?.fps ?? 0).toFixed(1)} fps`
            : replay === null && !analysed
              ? "Not analysed"
              : ""}
        </span>
      </div>
    </div>
  );
}
