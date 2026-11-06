import { useEffect, useState } from "react";
import { Box, Cctv, ChevronDown, LayoutGrid, Settings } from "lucide-react";
import { api, SERVER_UNREACHABLE, useLive, usePoll, type Status } from "./api";
import { SceneView } from "./components/SceneView";
import { DevicesView } from "./components/Devices";
import { Card, InfoColumn, VideoWall } from "./components/Views";
import { lastSceneMode, useRoute } from "./route";

type View = "scene" | "cameras" | "devices";

const NAV: { id: View; label: string; icon: typeof Box }[] = [
  { id: "scene", label: "3D Scene", icon: Box },
  { id: "cameras", label: "Cameras", icon: LayoutGrid },
  { id: "devices", label: "Devices", icon: Cctv },
];

function TopBar({ view, status }: { view: View; status: Status | null }) {
  const current = NAV.find((n) => n.id === view)!;
  const Icon = current.icon;
  return (
    <header className="relative flex h-[56px] shrink-0 items-stretch border-b border-line bg-bg">
      <div className="flex items-center gap-2.5 border-r border-line px-5">
        <span className={`h-1.5 w-1.5 rounded-full ${status ? "bg-ok" : "bg-faint"}`} />
        <span className="text-[14px] text-fg">Home</span>
        <ChevronDown size={14} className="text-muted" />
      </div>
      <div className="flex items-center gap-2 border-b-2 border-transparent bg-surface px-5 text-accent">
        <Icon size={18} strokeWidth={1.6} />
        <span className="text-[15px]">{current.label}</span>
      </div>
      <div className="pointer-events-none absolute inset-0 hidden items-center justify-center md:flex">
        <span className="text-[17px] font-semibold tracking-[0.02em] text-faint">Minastir</span>
      </div>
    </header>
  );
}

function SideRail({ view, setView }: { view: View; setView: (v: View) => void }) {
  return (
    <nav className="flex w-[64px] shrink-0 flex-col items-center gap-2 border-r border-line bg-bg py-4">
      {NAV.map(({ id, label, icon: Icon }) => (
        <button
          key={id}
          onClick={() => setView(id)}
          title={label}
          aria-label={label}
          className={`flex h-10 w-10 items-center justify-center rounded-md transition ${
            view === id ? "text-accent" : "text-muted hover:text-fg"
          }`}
        >
          <Icon size={22} strokeWidth={1.4} />
        </button>
      ))}
      <div className="my-2 h-px w-7 bg-line-strong" />
      <button title="Settings" aria-label="Settings" className="flex h-10 w-10 items-center justify-center text-muted hover:text-fg">
        <Settings size={22} strokeWidth={1.4} />
      </button>
    </nav>
  );
}

export default function App() {
  const { route, go } = useRoute();
  const view: View = route === "cameras" ? "cameras" : route === "devices" ? "devices" : "scene";
  // The rail's "3D Scene" opens the scene as it was last left (3D or plan).
  const setView = (v: View) => go(v === "scene" ? lastSceneMode() : v);
  const [selected, setSelected] = useState<string | null>(null);
  const cameras = usePoll(api.cameras);
  const status = usePoll(api.status, 2000);
  // One live connection for the whole app: the scene and the camera list both show it.
  const live = useLive();
  const [captureError, setCaptureError] = useState<string | null>(null);

  // Refresh camera list when a capture job finishes (has_background flags change).
  const running = status.data?.job.running;
  useEffect(() => {
    if (running === false) cameras.reload();
  }, [running, cameras.reload]);

  async function capture(what: "capture" | "calibrate" = "capture") {
    setCaptureError(null);
    try {
      await (what === "calibrate" ? api.calibrate() : api.capture());
      status.reload();
    } catch (e) {
      setCaptureError(e instanceof Error ? e.message : String(e));
    }
  }

  const list = cameras.data ?? [];
  const empty = cameras.data !== null && list.length === 0;

  // First run: nothing adopted yet, so the device list *is* the onboarding.
  const [routedFirstRun, setRoutedFirstRun] = useState(false);
  useEffect(() => {
    if (cameras.data === null || routedFirstRun) return;
    setRoutedFirstRun(true);
    if (cameras.data.length === 0) go("devices", true);
  }, [cameras.data, routedFirstRun]);
  const backendDown = cameras.error !== null && cameras.data === null;
  const connectionLost = status.error === SERVER_UNREACHABLE && !backendDown;

  // Reload the camera list whenever the connection (re)establishes.
  const online = status.data !== null && status.error === null;
  useEffect(() => {
    if (online) cameras.reload();
  }, [online, cameras.reload]);

  let body: React.ReactNode;
  if (backendDown) {
    body = (
      <Centered>
        <Card className="px-6 py-5">
          <h1 className="text-[16px] font-medium">Can't reach the Minastir server</h1>
          <p className="mt-1.5 text-[13px] text-muted">
            Start it with <code className="rounded bg-sunken px-1 py-0.5">npm run dev</code> from the repo root.
          </p>
        </Card>
      </Centered>
    );
  } else if (empty && view !== "devices") {
    body = (
      <Centered>
        <Card className="px-6 py-5">
          <h1 className="text-[16px] font-medium">No cameras adopted yet</h1>
          <p className="mt-1.5 text-[13px] text-muted">Adopt cameras from the device list to see them here.</p>
          <button
            onClick={() => setView("devices")}
            className="mt-4 h-8 rounded-md bg-accent px-3.5 text-[13px] font-medium text-white hover:bg-[#005ee0]"
          >
            Go to Devices
          </button>
        </Card>
      </Centered>
    );
  } else if (view === "devices") {
    body = <DevicesView cameras={list} onChanged={cameras.reload} />;
  } else {
    body = (
      <div className="flex h-full min-h-0 gap-4">
        <InfoColumn
          cameras={list}
          status={status.data}
          live={live.snapshot}
          selected={selected}
          onSelect={setSelected}
          onCapture={() => capture("capture")}
          onCalibrate={() => capture("calibrate")}
        />
        {view === "scene" ? (
          <SceneView
            cameras={list}
            live={live}
            selected={selected}
            onSelect={setSelected}
            onChanged={cameras.reload}
            mode={route === "plan" ? "plan" : "3d"}
            onModeChange={(m) => go(m)}
          />
        ) : (
          <VideoWall cameras={list} />
        )}
      </div>
    );
  }

  return (
    <div className="flex h-dvh flex-col">
      <TopBar view={view} status={status.data} />
      <div className="flex min-h-0 flex-1">
        <SideRail view={view} setView={setView} />
        <main className="relative min-w-0 flex-1 p-4 md:p-6 md:pb-4">
          {body}
          {connectionLost && (
            <div className="absolute left-1/2 top-3 z-20 flex -translate-x-1/2 items-center gap-2.5 rounded-md bg-surface px-4 py-2.5 text-[13px] shadow-[var(--shadow-float)]">
              <span className="h-2 w-2 animate-pulse rounded-full bg-warn" />
              Lost connection to the Minastir server — reconnecting…
            </div>
          )}
          {captureError && (
            <div
              className="absolute left-1/2 top-4 -translate-x-1/2 cursor-pointer rounded-md bg-surface px-4 py-2.5 text-[13px] text-bad shadow-[var(--shadow-float)]"
              onClick={() => setCaptureError(null)}
            >
              {captureError}
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

function Centered({ children }: { children: React.ReactNode }) {
  return (
    <div className="scroll-thin flex h-full items-center justify-center overflow-y-auto">
      <div className="w-full max-w-lg">{children}</div>
    </div>
  );
}
