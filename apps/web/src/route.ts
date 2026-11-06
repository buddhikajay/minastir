import { useCallback, useEffect, useRef, useState } from "react";

/** Pages with their own address, so a reload (or a rebuild in development) stays put. */
export type Route = "3d" | "plan" | "cameras" | "devices";
const ROUTES: Route[] = ["3d", "plan", "cameras", "devices"];

export function lastSceneMode(): Route {
  try {
    return localStorage.getItem("minastir.sceneMode") === "plan" ? "plan" : "3d";
  } catch {
    return "3d";
  }
}

function parse(path: string): Route {
  const p = path.replace(/^\/+|\/+$/g, "") as Route;
  // "/" opens the scene as it was last left (3D or plan).
  return ROUTES.includes(p) ? p : lastSceneMode();
}

/** Current page from the address bar; `go` changes it (adding a history entry, so Back works). */
export function useRoute() {
  const [route, setRoute] = useState<Route>(() => parse(window.location.pathname));
  useEffect(() => {
    const onPop = () => setRoute(parse(window.location.pathname));
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);
  useEffect(() => {
    if (route === "3d" || route === "plan") {
      try {
        localStorage.setItem("minastir.sceneMode", route);
      } catch {
        /* private mode */
      }
    }
    // Show the page's address even when we arrived at "/".
    if (window.location.pathname !== `/${route}`) window.history.replaceState(null, "", `/${route}${window.location.search}`);
  }, [route]);
  // History changes happen here, not inside a state updater (React may run those twice).
  const current = useRef(route);
  current.current = route;
  const go = useCallback((r: Route, replace = false) => {
    if (current.current === r) return;
    window.history[replace ? "replaceState" : "pushState"](null, "", `/${r}`);
    current.current = r;
    setRoute(r);
  }, []);
  return { route, go };
}
