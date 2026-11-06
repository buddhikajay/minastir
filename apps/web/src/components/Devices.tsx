import { useCallback, useEffect, useMemo, useState } from "react";
import {
  BellRing,
  Camera as CameraIcon,
  Cctv,
  ChevronUp,
  CircleHelp,
  LayoutGrid,
  List,
  Loader2,
  PanelLeftClose,
  PanelLeftOpen,
  Pencil,
  Plus,
  RefreshCw,
  Search,
  Webcam,
  X,
} from "lucide-react";
import { api, bestProfile, resolution, SERVER_UNREACHABLE, type Camera, type Discovered } from "../api";
import { LiveImage } from "./LiveImage";
import { RES_META, resClass, type ResClass } from "./Views";

type Status = "adopted" | "pending";

/** Unified row: an adopted camera or a detected-but-not-adopted one. */
type DeviceRow = {
  key: string;
  status: Status;
  name: string;
  model: string;
  ip: string;
  res: ResClass | null;
  camera?: Camera;
  device?: Discovered;
};

type Panel = { kind: "adopted"; id: string } | { kind: "detected"; device: Discovered } | { kind: "manual" } | null;

const input =
  "h-9 w-full rounded-md border border-line-strong bg-surface px-3 text-[14px] text-fg placeholder:text-faint outline-none transition focus:border-accent focus:ring-2 focus:ring-accent/15";

const STATUS_LABEL: Record<Status, string> = { adopted: "Adopted", pending: "Ready to adopt" };

/** "ReolinkDuoFloodlightPoE" → "Reolink Duo Floodlight PoE"; leaves model codes like "DS-2CD2143G2-I" alone. */
function humanize(s: string | null | undefined): string | null {
  if (!s) return null;
  if (!/^[A-Za-z]+$/.test(s)) return s;
  return s
    .replace(/PoE/g, "\u0000")
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replace(/\u0000/g, " PoE")
    .replace(/\s+/g, " ")
    .trim();
}

function detectedName(d: Discovered): string {
  // Server-picked real name first; else the discovery name unless it's a placeholder like "IPC-BO"
  // ("Tapo C100" is more descriptive than the bare "C100" hardware code); else the model.
  const scope = d.name && !/^IPC(-|_|$)/i.test(d.name) ? d.name : null;
  return d.display_name ?? scope ?? humanize(d.hardware) ?? "ONVIF camera";
}

function TypeIcon({ model }: { model: string }) {
  const m = model.toLowerCase();
  const Icon = m.includes("doorbell") ? BellRing : /tapo|c1\d\d|c2\d\d|indoor|cube/.test(m) ? Webcam : m.includes("ptz") ? CameraIcon : Cctv;
  return <Icon size={20} strokeWidth={1.4} className="text-muted" />;
}

function useDiscovery() {
  const [devices, setDevices] = useState<Discovered[] | null>(null);
  const [scanning, setScanning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scan = useCallback(async () => {
    setScanning(true);
    try {
      setDevices(await api.discover());
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setScanning(false);
    }
  }, []);
  useEffect(() => {
    scan();
    const t = setInterval(scan, 30_000);
    return () => clearInterval(t);
  }, [scan]);
  return { devices, scanning, error, scan };
}

// ---------- filter column ----------

function FilterSection({
  title,
  options,
  selected,
  onToggle,
}: {
  title: string;
  options: { value: string; label: string; count: number }[];
  selected: Set<string>;
  onToggle: (v: string) => void;
}) {
  const [open, setOpen] = useState(true);
  if (!options.length) return null;
  return (
    <div className="border-b border-line py-3.5 last:border-0">
      <button onClick={() => setOpen(!open)} className="flex w-full items-center justify-between text-[14px] font-medium">
        {title}
        <ChevronUp size={16} className={`text-muted transition ${open ? "" : "rotate-180"}`} />
      </button>
      {open && (
        <div className="mt-2.5 flex flex-col gap-2">
          {options.map((o) => (
            <label key={o.value} className="flex cursor-pointer items-center gap-2.5 text-[14px]">
              <input
                type="checkbox"
                checked={selected.has(o.value)}
                onChange={() => onToggle(o.value)}
                className="h-4 w-4 rounded border-line-strong accent-accent"
              />
              <span className="truncate">{o.label}</span>
              <span className="text-muted">({o.count})</span>
            </label>
          ))}
        </div>
      )}
    </div>
  );
}

function facet<T extends string>(rows: DeviceRow[], get: (r: DeviceRow) => T | null, label: (v: T) => string) {
  const counts = new Map<T, number>();
  rows.forEach((r) => {
    const v = get(r);
    if (v) counts.set(v, (counts.get(v) ?? 0) + 1);
  });
  return [...counts.entries()]
    .sort((a, b) => label(a[0]).localeCompare(label(b[0])))
    .map(([value, count]) => ({ value, label: label(value), count }));
}

function useToggleSet() {
  const [set, setSet] = useState<Set<string>>(new Set());
  const toggle = (v: string) =>
    setSet((s) => {
      const n = new Set(s);
      n.has(v) ? n.delete(v) : n.add(v);
      return n;
    });
  return [set, toggle] as const;
}

// ---------- page ----------

export function DevicesView({ cameras, onChanged }: { cameras: Camera[]; onChanged: () => void }) {
  const discovery = useDiscovery();
  const [panel, setPanel] = useState<Panel>(null);
  const [layout, setLayout] = useState<"list" | "grid">("list");
  const [filtersOpen, setFiltersOpen] = useState(true);
  const [query, setQuery] = useState("");
  const [models, toggleModel] = useToggleSet();
  const [statuses, toggleStatus] = useToggleSet();
  const [resolutions, toggleRes] = useToggleSet();

  const rows: DeviceRow[] = useMemo(() => {
    const pending = (discovery.devices ?? []).filter(
      (d) => !d.adopted_id && !cameras.some((c) => c.device_url === d.device_url),
    );
    return [
      ...cameras.map((c) => ({
        key: c.id,
        status: "adopted" as const,
        name: c.label,
        model: c.model ?? "—",
        ip: c.ip,
        res: resClass(c),
        camera: c,
      })),
      ...pending.map((d) => ({
        key: d.endpoint,
        status: "pending" as const,
        name: detectedName(d),
        model: humanize(d.hardware) ?? "—",
        ip: d.ip,
        res: null,
        device: d,
      })),
    ];
  }, [cameras, discovery.devices]);

  const q = query.trim().toLowerCase();
  const visible = rows.filter(
    (r) =>
      (!q || `${r.name} ${r.model} ${r.ip}`.toLowerCase().includes(q)) &&
      (!models.size || models.has(r.model)) &&
      (!statuses.size || statuses.has(r.status)) &&
      (!resolutions.size || (r.res && resolutions.has(r.res))),
  );
  const pendingCount = rows.filter((r) => r.status === "pending").length;

  const open = (r: DeviceRow) =>
    setPanel(r.camera ? { kind: "adopted", id: r.camera.id } : { kind: "detected", device: r.device! });
  const isActive = (r: DeviceRow) =>
    (panel?.kind === "adopted" && r.camera?.id === panel.id) ||
    (panel?.kind === "detected" && r.device?.endpoint === panel.device.endpoint);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setPanel(null);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  async function afterAdopt(cam: Camera) {
    onChanged();
    await discovery.scan();
    setPanel({ kind: "adopted", id: cam.id });
  }

  const th = "whitespace-nowrap px-4 py-3 text-[14px] font-medium text-fg";
  const td = "whitespace-nowrap px-4 py-3";

  return (
    <div className="relative flex h-full min-h-0 gap-4">
      {/* filters */}
      {filtersOpen && (
        <aside className="scroll-thin hidden w-[280px] shrink-0 flex-col overflow-y-auto lg:flex">
          <div className="flex items-center gap-2">
            <div className="relative flex-1">
              <Search size={16} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-faint" />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search"
                className="h-10 w-full rounded-md bg-sunken pl-9 pr-3 text-[14px] outline-none placeholder:text-faint focus:ring-2 focus:ring-accent/15"
              />
            </div>
            <div className="flex h-10 rounded-md bg-sunken p-1">
              {(["list", "grid"] as const).map((l) => {
                const Icon = l === "list" ? List : LayoutGrid;
                return (
                  <button
                    key={l}
                    onClick={() => setLayout(l)}
                    aria-label={`${l} view`}
                    className={`flex w-9 items-center justify-center rounded ${
                      layout === l ? "bg-surface text-accent shadow-sm" : "text-muted hover:text-fg"
                    }`}
                  >
                    <Icon size={17} strokeWidth={1.6} />
                  </button>
                );
              })}
            </div>
            <span title="Cameras are found with ONVIF WS-Discovery on your local network." className="text-muted">
              <CircleHelp size={20} strokeWidth={1.4} />
            </span>
          </div>

          <div className="mt-2">
            <FilterSection
              title="Status"
              options={facet(rows, (r) => r.status, (v) => STATUS_LABEL[v])}
              selected={statuses}
              onToggle={toggleStatus}
            />
            <FilterSection title="Model" options={facet(rows, (r) => r.model, (v) => v)} selected={models} onToggle={toggleModel} />
            <FilterSection
              title="Recording Resolution"
              options={facet(rows, (r) => r.res, (v) => RES_META[v].label)}
              selected={resolutions}
              onToggle={toggleRes}
            />
          </div>
        </aside>
      )}

      {/* table */}
      <section className="flex min-w-0 flex-1 flex-col overflow-hidden rounded-[10px] border border-line bg-surface">
        <div className="flex items-center gap-3 border-b border-line px-4 py-2.5">
          <button
            onClick={() => setFiltersOpen(!filtersOpen)}
            className="hidden rounded-md p-1.5 text-muted hover:bg-bg hover:text-fg lg:block"
            aria-label={filtersOpen ? "Hide filters" : "Show filters"}
          >
            {filtersOpen ? <PanelLeftClose size={17} strokeWidth={1.5} /> : <PanelLeftOpen size={17} strokeWidth={1.5} />}
          </button>
          <span className="min-w-0 truncate whitespace-nowrap text-[13px] text-muted">
            {cameras.length} adopted{pendingCount > 0 && ` · ${pendingCount} ready to adopt`}
          </span>
          <div className="ml-auto flex shrink-0 items-center gap-2 whitespace-nowrap">
            <button
              onClick={discovery.scan}
              disabled={discovery.scanning}
              className="flex h-8 items-center gap-1.5 rounded-md px-2.5 text-[13px] text-muted transition hover:bg-bg hover:text-fg disabled:opacity-60"
            >
              <RefreshCw size={14} className={discovery.scanning ? "animate-spin" : ""} />
              {discovery.scanning ? "Scanning…" : "Scan"}
            </button>
            <button
              onClick={() => setPanel({ kind: "manual" })}
              className="flex h-8 items-center gap-1.5 rounded-md bg-accent px-3 text-[13px] font-medium text-white transition hover:bg-[#005ee0]"
            >
              <Plus size={14} /> Add camera
            </button>
          </div>
        </div>

        <div className="scroll-thin min-h-0 flex-1 overflow-auto">
          {layout === "list" ? (
            <table className="w-full text-left text-[14px]">
              <thead className="sticky top-0 z-[1] bg-surface">
                <tr className="border-b border-line">
                  <th className="w-8" />
                  <th className={th}>Type</th>
                  <th className={th}>Name</th>
                  <th className={th}>Status</th>
                  <th className={th}>Model</th>
                  <th className={th}>IP Address</th>
                  <th className={th}>Recording Res.</th>
                </tr>
              </thead>
              <tbody>
                {visible.map((r) => (
                  <tr
                    key={r.key}
                    onClick={() => open(r)}
                    className={`cursor-pointer border-b border-line transition ${isActive(r) ? "bg-accent/[0.06]" : "hover:bg-bg"}`}
                  >
                    <td className="pl-4">
                      <span className={`block h-2 w-2 rounded-full ${r.status === "adopted" ? "bg-ok" : "bg-accent"}`} />
                    </td>
                    <td className="px-4 py-2">
                      <TypeIcon model={`${r.model} ${r.name}`} />
                    </td>
                    <td className={`${td} max-w-[220px] truncate`} title={r.name}>
                      {r.name}
                    </td>
                    <td className={`${td} ${r.status === "pending" ? "text-accent" : ""}`}>
                      {r.status === "adopted" ? "Adopted" : "Ready to adopt"}
                    </td>
                    <td className={`${td} max-w-[220px] truncate`} title={r.model}>
                      {r.model}
                    </td>
                    <td className={`${td} font-mono text-[13px]`}>{r.ip}</td>
                    <td className={td}>{r.res ? RES_META[r.res].label : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div className="grid grid-cols-[repeat(auto-fill,minmax(220px,1fr))] gap-3 p-4">
              {visible.map((r) => (
                <button
                  key={r.key}
                  onClick={() => open(r)}
                  className={`overflow-hidden rounded-md border text-left transition ${
                    isActive(r) ? "border-accent" : "border-line hover:border-line-strong"
                  }`}
                >
                  {r.camera ? (
                    <LiveImage src={api.snapshotUrl(r.camera.id)} interval={8000} className="aspect-video w-full" />
                  ) : (
                    <div className="flex aspect-video w-full items-center justify-center bg-sunken">
                      <TypeIcon model={`${r.model} ${r.name}`} />
                    </div>
                  )}
                  <div className="flex items-center gap-2 px-3 py-2.5">
                    <span className={`h-2 w-2 shrink-0 rounded-full ${r.status === "adopted" ? "bg-ok" : "bg-accent"}`} />
                    <span className="truncate text-[13px] font-medium">{r.name}</span>
                    <span className="ml-auto shrink-0 font-mono text-[11px] text-faint">{r.ip}</span>
                  </div>
                </button>
              ))}
            </div>
          )}

          {visible.length === 0 && (
            <div className="px-6 py-16 text-center">
              {rows.length > 0 ? (
                <span className="text-[14px] text-muted">No devices match the filters</span>
              ) : discovery.scanning || (discovery.devices === null && !discovery.error) ? (
                <span className="inline-flex items-center gap-2 text-[14px] text-muted">
                  <Loader2 size={15} className="animate-spin" /> Looking for cameras on your network…
                </span>
              ) : (
                <div className="mx-auto max-w-sm">
                  <div className="text-[15px]">No cameras detected</div>
                  <p className="mt-1.5 text-[13px] leading-relaxed text-muted">
                    Some cameras ship with ONVIF turned off, or sit on another network segment. Add them by IP address
                    or by their RTSP stream URL instead.
                  </p>
                  <button onClick={() => setPanel({ kind: "manual" })} className="mt-3 text-[13px] font-medium text-accent hover:underline">
                    Add camera
                  </button>
                </div>
              )}
            </div>
          )}
          {discovery.error && discovery.error !== SERVER_UNREACHABLE && (
            <p className="px-4 py-3 text-[12px] text-bad">Discovery failed: {discovery.error}</p>
          )}
        </div>
      </section>

      {panel && (
        <aside className="scroll-thin absolute bottom-0 right-0 top-0 z-20 w-[min(400px,100%)] overflow-y-auto rounded-[10px] border border-line bg-surface shadow-[0_8px_40px_rgb(16_24_40/0.16)] animate-[drawer-in_180ms_ease-out]">
          {panel.kind === "detected" && (
            <AdoptPanel
              key={panel.device.endpoint}
              title={detectedName(panel.device)}
              subtitle={[humanize(panel.device.hardware), panel.device.ip].filter(Boolean).join(" · ")}
              target={{ device_url: panel.device.device_url }}
              onClose={() => setPanel(null)}
              onAdopted={afterAdopt}
            />
          )}
          {panel.kind === "manual" && (
            <AddCameraPanel onClose={() => setPanel(null)} onAdded={afterAdopt} />
          )}
          {panel.kind === "adopted" &&
            (() => {
              const cam = cameras.find((c) => c.id === panel.id);
              return cam ? (
                <CameraPanel
                  key={cam.id}
                  camera={cam}
                  onClose={() => setPanel(null)}
                  onUpdated={afterAdopt}
                  onRemoved={() => {
                    setPanel(null);
                    onChanged();
                    discovery.scan();
                  }}
                />
              ) : null;
            })()}
        </aside>
      )}
    </div>
  );
}

function PanelHeader({ title, subtitle, onClose }: { title: string; subtitle?: string; onClose: () => void }) {
  return (
    <div className="flex items-start gap-3 border-b border-line px-5 py-4">
      <div className="min-w-0 flex-1">
        <div className="truncate text-[15px] font-medium">{title}</div>
        {subtitle && <div className="mt-0.5 truncate text-[12px] text-muted">{subtitle}</div>}
      </div>
      <button onClick={onClose} className="-mr-1.5 rounded-md p-1.5 text-muted hover:bg-bg hover:text-fg" aria-label="Close">
        <X size={16} />
      </button>
    </div>
  );
}

/** Username/password (and optionally address) form that logs in and adopts the camera. */
function CredentialsForm({
  target,
  manual = false,
  submitLabel,
  onAdopted,
}: {
  target?: { device_url: string };
  manual?: boolean;
  submitLabel: string;
  onAdopted: (c: Camera) => void;
}) {
  const [host, setHost] = useState("");
  const [user, setUser] = useState("admin");
  const [pass, setPass] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      onAdopted(await api.adopt(manual ? { host } : target!, user, pass));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit} className="flex flex-col gap-3.5">
      {manual && (
        <label className="flex flex-col gap-1.5">
          <span className="text-[12px] text-muted">IP address</span>
          <input
            className={input}
            value={host}
            onChange={(e) => setHost(e.target.value)}
            placeholder="192.168.1.20 or 192.168.1.20:8000"
            autoFocus
            required
          />
        </label>
      )}
      <label className="flex flex-col gap-1.5">
        <span className="text-[12px] text-muted">Username</span>
        <input className={input} value={user} onChange={(e) => setUser(e.target.value)} autoComplete="username" required />
      </label>
      <label className="flex flex-col gap-1.5">
        <span className="text-[12px] text-muted">Password</span>
        <input
          className={input}
          type="password"
          value={pass}
          onChange={(e) => setPass(e.target.value)}
          autoComplete="current-password"
          autoFocus={!manual}
        />
      </label>
      {error && <p className="rounded-md bg-bad/[0.07] px-3 py-2 text-[13px] text-bad">{error}</p>}
      <button
        disabled={busy}
        className="mt-1 flex h-9 items-center justify-center gap-2 rounded-md bg-accent text-[13px] font-medium text-white transition hover:bg-[#005ee0] disabled:opacity-60"
      >
        {busy && <Loader2 size={14} className="animate-spin" />}
        {busy ? "Connecting…" : submitLabel}
      </button>
    </form>
  );
}

function AdoptPanel({
  title,
  subtitle,
  target,
  onClose,
  onAdopted,
}: {
  title: string;
  subtitle?: string;
  target: { device_url: string };
  onClose: () => void;
  onAdopted: (c: Camera) => void;
}) {
  return (
    <>
      <PanelHeader title={title} subtitle={subtitle} onClose={onClose} />
      <div className="px-5 py-4">
        <p className="mb-4 text-[13px] leading-relaxed text-muted">
          Enter this camera's ONVIF login. It's usually the same account you use in the camera's own app or web page.
        </p>
        <CredentialsForm target={target} submitLabel="Adopt" onAdopted={onAdopted} />
      </div>
    </>
  );
}

/** Manual add: by IP over ONVIF, or by a raw RTSP/RTSPS URL (UniFi Protect, NVRs, non-ONVIF cameras). */
function AddCameraPanel({ onClose, onAdded }: { onClose: () => void; onAdded: (c: Camera) => void }) {
  const [mode, setMode] = useState<"onvif" | "stream">("onvif");
  const tab = (m: typeof mode, label: string) => (
    <button
      type="button"
      onClick={() => setMode(m)}
      className={`h-7 flex-1 rounded-[5px] text-[13px] transition ${mode === m ? "bg-surface font-medium shadow-sm" : "text-muted hover:text-fg"}`}
    >
      {label}
    </button>
  );
  return (
    <>
      <PanelHeader title="Add camera" onClose={onClose} />
      <div className="px-5 py-4">
        <div className="mb-4 flex rounded-md bg-sunken p-1">
          {tab("onvif", "ONVIF camera")}
          {tab("stream", "Stream URL")}
        </div>
        {mode === "onvif" ? (
          <>
            <p className="mb-4 text-[13px] leading-relaxed text-muted">
              Enter the camera's address and its ONVIF login. Minastir will read its streams and add it.
            </p>
            <CredentialsForm manual submitLabel="Adopt" onAdopted={onAdded} />
          </>
        ) : (
          <StreamForm onAdded={onAdded} />
        )}
      </div>
    </>
  );
}

function StreamForm({ onAdded }: { onAdded: (c: Camera) => void }) {
  const [url, setUrl] = useState("");
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const unifi = /:744[17]\//.test(url);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      onAdded(await api.addStream(url, name));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit} className="flex flex-col gap-3.5">
      <p className="text-[13px] leading-relaxed text-muted">
        For cameras without ONVIF. In UniFi Protect, open the camera's settings → <b className="font-medium text-fg">Share Livestream</b> →{" "}
        <b className="font-medium text-fg">Enable Secure RTSPS Output</b>, tick the highest resolution, and copy the link.
      </p>
      <label className="flex flex-col gap-1.5">
        <span className="text-[12px] text-muted">Stream URL</span>
        <input
          className={`${input} font-mono text-[12px]`}
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="rtsps://192.168.1.1:7441/…"
          autoFocus
          required
          spellCheck={false}
        />
      </label>
      <label className="flex flex-col gap-1.5">
        <span className="text-[12px] text-muted">Name</span>
        <input
          className={input}
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder={unifi ? "As in UniFi Protect, e.g. Back Yard" : "e.g. Back Yard"}
        />
        {unifi && (
          <span className="text-[12px] text-faint">UniFi stream links don't carry the camera's name, so type it here.</span>
        )}
      </label>
      {error && <p className="rounded-md bg-bad/[0.07] px-3 py-2 text-[13px] text-bad">{error}</p>}
      <button
        disabled={busy}
        className="mt-1 flex h-9 items-center justify-center gap-2 rounded-md bg-accent text-[13px] font-medium text-white transition hover:bg-[#005ee0] disabled:opacity-60"
      >
        {busy && <Loader2 size={14} className="animate-spin" />}
        {busy ? "Opening stream…" : "Add stream"}
      </button>
    </form>
  );
}

/** Stream URL with its secret part (UniFi token / path) masked. */
function maskedUrl(u: string): string {
  const m = u.match(/^(\w+:\/\/[^/]+)(\/.*)?$/);
  return m ? `${m[1]}/${m[2] && m[2].length > 1 ? "••••••" : ""}` : u;
}

/** Native aspect of the recording stream — panoramic/dual-lens cameras are far wider than 16:9. */
function aspectOf(c: Camera): string {
  const p = bestProfile(c);
  return p?.width && p?.height ? `${p.width} / ${p.height}` : "16 / 9";
}

/** Panel title that doubles as the rename control (click the name or the pencil). */
function NameHeader({ camera, onClose, onSaved }: { camera: Camera; onClose: () => void; onSaved: (c: Camera) => void }) {
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState(camera.label);
  const [saving, setSaving] = useState(false);
  const fallback = camera.detected_name ?? camera.model_label;

  async function save(name: string | null) {
    setSaving(true);
    try {
      onSaved(await api.rename(camera.id, name));
      setEditing(false);
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="flex items-start gap-3 border-b border-line px-5 py-4">
      <div className="min-w-0 flex-1">
        {editing ? (
          <form
            onSubmit={(e) => {
              e.preventDefault();
              save(value.trim() || null);
            }}
            className="flex items-center gap-2"
          >
            <input
              autoFocus
              value={value}
              onChange={(e) => setValue(e.target.value)}
              onKeyDown={(e) => e.key === "Escape" && (e.stopPropagation(), setEditing(false))}
              placeholder={fallback}
              className="h-8 min-w-0 flex-1 rounded-md border border-accent px-2 text-[15px] font-medium outline-none ring-2 ring-accent/15"
            />
            <button disabled={saving} className="h-8 rounded-md bg-accent px-3 text-[13px] font-medium text-white disabled:opacity-60">
              Save
            </button>
          </form>
        ) : (
          <button
            onClick={() => {
              setValue(camera.name ?? camera.label);
              setEditing(true);
            }}
            className="group flex max-w-full items-center gap-2 text-left"
            title="Rename"
          >
            <span className="truncate text-[15px] font-medium">{camera.label}</span>
            <Pencil size={13} className="shrink-0 text-faint transition group-hover:text-accent" />
          </button>
        )}
        <div className="mt-0.5 truncate text-[12px] text-muted">
          {camera.ip}
          {camera.name && (
            <>
              {" · "}
              <button onClick={() => save(null)} className="hover:text-accent" title={`Use "${fallback}"`}>
                Reset to detected name
              </button>
            </>
          )}
        </div>
      </div>
      <button onClick={onClose} className="-mr-1.5 rounded-md p-1.5 text-muted hover:bg-bg hover:text-fg" aria-label="Close">
        <X size={16} />
      </button>
    </div>
  );
}

function Detail({ label, value, mono = false }: { label: string; value: React.ReactNode; mono?: boolean }) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-[6px] text-[13px]">
      <span className="shrink-0 text-muted">{label}</span>
      <span className={`truncate text-right ${mono ? "font-mono text-[12px]" : ""}`}>{value}</span>
    </div>
  );
}

function CameraPanel({
  camera,
  onClose,
  onUpdated,
  onRemoved,
}: {
  camera: Camera;
  onClose: () => void;
  onUpdated: (c: Camera) => void;
  onRemoved: () => void;
}) {
  const [editing, setEditing] = useState(false);
  const [confirmRemove, setConfirmRemove] = useState(false);
  const [removing, setRemoving] = useState(false);
  const [refreshing, setRefreshing] = useState(false);

  async function refresh() {
    setRefreshing(true);
    try {
      onUpdated(await api.refresh(camera.id));
    } finally {
      setRefreshing(false);
    }
  }

  async function remove() {
    setRemoving(true);
    try {
      await api.remove(camera.id);
      onRemoved();
    } finally {
      setRemoving(false);
    }
  }

  return (
    <>
      <NameHeader camera={camera} onClose={onClose} onSaved={onUpdated} />
      <LiveImage
        src={api.snapshotUrl(camera.id)}
        interval={2000}
        className="w-full bg-video"
        style={{ aspectRatio: aspectOf(camera) }}
      />
      <div className="px-5 py-3">
        <Detail label="Status" value={<span className="text-ok">Connected</span>} />
        {camera.kind === "stream" ? (
          <>
            <Detail label="Source" value={camera.model ?? "RTSP stream"} />
            <Detail label="Stream" value={maskedUrl(camera.device_url)} mono />
          </>
        ) : (
          <>
            <Detail label="Manufacturer" value={camera.manufacturer ?? "—"} />
            <Detail label="Model" value={camera.model ?? "—"} />
            <Detail label="Firmware" value={camera.firmware ?? "—"} />
            <Detail label="Serial" value={camera.serial ?? "—"} mono />
            <Detail label="PTZ" value={camera.ptz ? "Supported" : "No"} />
          </>
        )}
        <Detail label="Recording stream" value={`${resolution(camera)} ${bestProfile(camera)?.encoding ?? ""}`} />
      </div>

      <div className="border-t border-line px-5 py-3">
        <div className="pb-1 text-[12px] text-muted">Streams</div>
        {camera.profiles.map((p) => (
          <div key={p.token} className="flex items-center justify-between py-[5px] text-[13px]">
            <span>{p.name}</span>
            <span className="text-muted">
              {p.width && p.height ? `${p.width}×${p.height}` : "—"} {p.encoding ?? ""}
            </span>
          </div>
        ))}
      </div>

      <div className="border-t border-line px-5 py-4">
        {editing ? (
          <>
            <div className="mb-3 flex items-center justify-between">
              <span className="text-[13px] font-medium">Update credentials</span>
              <button onClick={() => setEditing(false)} className="text-[12px] text-muted hover:text-fg">
                Cancel
              </button>
            </div>
            <CredentialsForm
              target={{ device_url: camera.device_url }}
              submitLabel="Save and reconnect"
              onAdopted={(c) => {
                setEditing(false);
                onUpdated(c);
              }}
            />
          </>
        ) : (
          <div className="flex items-center gap-2">
            {camera.kind === "onvif" && (
              <button
                onClick={() => setEditing(true)}
                className="h-8 rounded-md border border-line-strong px-3 text-[13px] transition hover:bg-bg"
              >
                Update credentials
              </button>
            )}
            <button
              onClick={refresh}
              disabled={refreshing}
              title="Re-read name, streams and firmware from the camera"
              className="flex h-8 items-center gap-1.5 rounded-md border border-line-strong px-3 text-[13px] transition hover:bg-bg disabled:opacity-60"
            >
              <RefreshCw size={13} className={refreshing ? "animate-spin" : ""} /> Refresh
            </button>
            {confirmRemove ? (
              <button
                onClick={remove}
                disabled={removing}
                className="ml-auto h-8 rounded-md bg-bad px-3 text-[13px] font-medium text-white disabled:opacity-60"
              >
                {removing ? "Removing…" : "Confirm remove"}
              </button>
            ) : (
              <button
                onClick={() => setConfirmRemove(true)}
                className="ml-auto h-8 rounded-md px-3 text-[13px] text-bad transition hover:bg-bad/[0.07]"
              >
                Remove
              </button>
            )}
          </div>
        )}
      </div>
    </>
  );
}
