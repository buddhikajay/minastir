import { useEffect, useState } from "react";
import { Loader2, Move3d, X } from "lucide-react";
import { api, bestProfile, resolution, type Camera, type LiveSnapshot, type Status } from "../api";
import { CameraList } from "./CameraList";
import { LiveImage } from "./LiveImage";

// ---------- shared bits ----------

export function Card({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <section className={`rounded-[10px] border border-line bg-surface ${className}`}>{children}</section>;
}

function Row({ label, value, tone }: { label: string; value: React.ReactNode; tone?: "muted" | "ok" }) {
  return (
    <div className="flex items-center justify-between py-[7px] text-[14px]">
      <span className="text-fg">{label}</span>
      <span className={tone === "ok" ? "text-ok" : "text-muted"}>{value}</span>
    </div>
  );
}

export type ResClass = "hd" | "2k" | "4k";
export function resClass(c: Camera): ResClass {
  const p = bestProfile(c);
  const px = (p?.width ?? 0) * (p?.height ?? 0);
  if (px <= 1920 * 1080) return "hd";
  if (px <= 2688 * 1520) return "2k";
  return "4k";
}
export const RES_META: Record<ResClass, { label: string; bar: string; chip: string }> = {
  hd: { label: "HD", bar: "bg-hd", chip: "bg-hd" },
  "2k": { label: "2K", bar: "bg-2k", chip: "bg-2k" },
  "4k": { label: "4K/5K", bar: "bg-4k", chip: "bg-4k" },
};

function useClock() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(t);
  }, []);
  return now;
}

// ---------- info column (left) ----------

export function InfoColumn({
  cameras,
  status,
  live,
  selected,
  onSelect,
  onCapture,
  onCalibrate,
}: {
  cameras: Camera[];
  status: Status | null;
  live: LiveSnapshot | null;
  selected: string | null;
  onSelect: (id: string | null) => void;
  onCapture: () => void;
  onCalibrate: () => void;
}) {
  const calibrated = cameras.filter((c) => c.calibration).length;
  const now = useClock();
  const job = status?.job;
  const classes = (["hd", "2k", "4k"] as ResClass[]).map((k) => ({
    k,
    n: cameras.filter((c) => resClass(c) === k).length,
  }));

  return (
    <aside className="scroll-thin hidden w-[340px] shrink-0 flex-col gap-3 overflow-y-auto pb-4 lg:flex">
      <Card className="px-5 pb-4 pt-4">
        <div className="border-b border-line pb-3.5 text-center text-[15px] text-muted">
          {now.toLocaleDateString(undefined, { weekday: "short", day: "numeric", month: "short" })} /{" "}
          {now.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
        </div>
        <div className="pt-2.5">
          <Row label="Cameras online" value={`${cameras.length} / ${cameras.length}`} />
          <Row
            label="Scene coverage"
            value={status ? `${status.captured} / ${status.cameras} captured` : "—"}
            tone={status && status.cameras > 0 && status.captured === status.cameras ? "ok" : "muted"}
          />
          <Row
            label="Calibration"
            value={`${calibrated} / ${cameras.length} measured`}
            tone={cameras.length > 0 && calibrated === cameras.length ? "ok" : "muted"}
          />
          <Row label="Camera capacity" value={cameras.length} />
        </div>
        <div className="mt-2 flex h-[7px] gap-[2px] overflow-hidden rounded-full bg-sunken">
          {cameras
            .slice()
            .sort((a, b) => ["hd", "2k", "4k"].indexOf(resClass(a)) - ["hd", "2k", "4k"].indexOf(resClass(b)))
            .map((c) => (
              <div key={c.id} className={`flex-1 ${RES_META[resClass(c)].bar}`} />
            ))}
        </div>
        <div className="mt-3 flex items-center justify-between text-[12px]">
          {classes.map(({ k, n }) => (
            <span key={k} className="flex items-center gap-1.5 text-muted">
              <span className={`rounded-[4px] px-1.5 py-0.5 text-[11px] font-semibold text-white ${RES_META[k].chip}`}>
                {RES_META[k].label}
              </span>
              ×{n}
            </span>
          ))}
        </div>
      </Card>

      <Card className="px-5 py-4">
        <div className="flex items-center justify-between">
          <span className="text-[14px] font-medium">3D Scene</span>
          {job?.running && (
            <span className="text-[12px] text-muted">
              {job.done}/{job.total}
            </span>
          )}
        </div>
        <p className="mt-1.5 text-[13px] leading-relaxed text-muted">
          {job?.running
            ? job.message
            : job?.message && job.message !== "Capture complete" && job.message !== "Calibration complete"
              ? job.message
              : calibrated < cameras.length
                ? "Measure each camera's lens, tilt and height from its own view, so people and cars land in the right place."
                : "Cameras measured. Recalibrate after moving a camera; recapture if the view has changed."}
        </p>
        {job?.running ? (
          <div className="mt-3 h-[3px] overflow-hidden rounded-full bg-sunken">
            <div
              className="h-full bg-accent transition-[width] duration-500"
              style={{ width: `${(job.done / Math.max(job.total, 1)) * 100}%` }}
            />
          </div>
        ) : (
          <div className="mt-3 flex gap-2">
            <button
              onClick={onCalibrate}
              disabled={!cameras.length}
              className="h-8 rounded-md bg-accent px-3.5 text-[13px] font-medium text-white transition hover:bg-[#005ee0] disabled:opacity-40"
            >
              {calibrated ? "Recalibrate" : "Calibrate cameras"}
            </button>
            <button
              onClick={onCapture}
              disabled={!cameras.length}
              title="Grab a fresh people-free frame from every camera"
              className="h-8 rounded-md border border-line-strong px-3 text-[13px] transition hover:bg-bg disabled:opacity-40"
            >
              Recapture
            </button>
          </div>
        )}
      </Card>

      <CameraList cameras={cameras} live={live} selected={selected} onSelect={onSelect} />
    </aside>
  );
}

// ---------- selected camera card (floats over the scene, Tesla media-card style) ----------

export function SelectedCard({ camera, onClose }: { camera: Camera; onClose: () => void }) {
  return (
    <div className="absolute bottom-4 right-4 w-[min(320px,calc(100%-32px))] overflow-hidden rounded-[10px] bg-surface shadow-[var(--shadow-float)]">
      <LiveImage src={api.snapshotUrl(camera.id)} interval={1500} className="aspect-video w-full" />
      <div className="flex items-center gap-3 px-4 py-3">
        <div className="min-w-0 flex-1">
          <div className="truncate text-[14px] font-medium">{camera.label}</div>
          <div className="text-[12px] text-muted">
            {camera.ip} · {resolution(camera)} · {bestProfile(camera)?.encoding ?? ""}
          </div>
        </div>
        <button onClick={onClose} className="rounded-md p-1.5 text-muted hover:bg-bg hover:text-fg" aria-label="Close">
          <X size={16} />
        </button>
      </div>
    </div>
  );
}

// ---------- video wall ----------

function Timestamp() {
  const now = useClock();
  const pad = (n: number) => String(n).padStart(2, "0");
  return (
    <span className="video-text font-mono text-[11px] text-white/90">
      {now.getFullYear()}-{pad(now.getMonth() + 1)}-{pad(now.getDate())} {pad(now.getHours())}:{pad(now.getMinutes())}:
      {pad(now.getSeconds())}
    </span>
  );
}

export function VideoWall({ cameras }: { cameras: Camera[] }) {
  const [open, setOpen] = useState<Camera | null>(null);
  const n = cameras.length;
  const cols = n <= 1 ? 1 : n <= 4 ? 2 : n <= 9 ? 3 : 4;

  return (
    <div className="flex h-full min-h-0 flex-1 items-center overflow-hidden rounded-[10px] bg-video">
      <div className="grid w-full" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}>
        {cameras.map((c) => (
          <button key={c.id} onClick={() => setOpen(c)} className="group relative aspect-video overflow-hidden">
            <LiveImage src={api.snapshotUrl(c.id)} className="h-full w-full" alt={c.label} />
            <div className="pointer-events-none absolute left-2.5 top-2">
              <Timestamp />
            </div>
            <div className="pointer-events-none absolute bottom-2 left-2.5 right-2.5 flex items-center gap-1.5 opacity-0 transition group-hover:opacity-100">
              <span className="video-text truncate text-[12px] font-medium text-white">{c.label}</span>
              {c.ptz && <Move3d size={12} className="ml-auto text-white/80" />}
            </div>
          </button>
        ))}
      </div>

      {open && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-6" onClick={() => setOpen(null)}>
          <div className="w-full max-w-6xl" onClick={(e) => e.stopPropagation()}>
            <div className="relative overflow-hidden rounded-[10px] bg-video">
              <LiveImage src={api.snapshotUrl(open.id)} interval={1200} className="aspect-video w-full" />
              <div className="absolute left-3 top-2.5">
                <Timestamp />
              </div>
            </div>
            <div className="mt-3 flex items-center gap-3 text-white">
              <span className="text-[15px] font-medium">{open.label}</span>
              <span className="text-[13px] text-white/50">
                {open.ip} · {resolution(open)}
              </span>
              <button
                onClick={() => setOpen(null)}
                className="ml-auto rounded-md p-1.5 text-white/60 hover:bg-white/10 hover:text-white"
                aria-label="Close"
              >
                <X size={18} />
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export function Spinner() {
  return <Loader2 size={16} className="animate-spin text-muted" />;
}
