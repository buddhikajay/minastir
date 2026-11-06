/**
 * Hero scene: the Minastir console's live 3D map (brand/reference/Minastir-Console.html), built
 * from the shared object library — a house and its street at night: sculpted cars, a person in a
 * pool of light, clustered trees, cameras with their view fans and frustums, and two blind spots
 * named in red. Night mode, slow orbit.
 *
 * World axes: x east, y north, h up → three.js (x, h, −y).
 */
import * as THREE from "three";
import {
  PALETTE,
  applyMode,
  blob,
  createMaterials,
  makeCameraMount,
  makeCar,
  makeHouse,
  makePerson,
  makePool,
  makeTree,
  softBox,
  studioEnvironment,
} from "@minastir/objects";

const W = (x: number, y: number, h = 0) => new THREE.Vector3(x, h, -y);
const D = THREE.MathUtils.degToRad;

type Cam = { name: string; pos: [number, number, number]; hd: number };
const CAMS: Cam[] = [
  { name: "Front Yard (Bullet)", pos: [-12, -1, 2.8], hd: -20 },
  { name: "Front Yard (Turret)", pos: [-2, 4, 3.0], hd: -60 },
  { name: "Duo Floodlight", pos: [9, 7, 2.6], hd: -120 },
  { name: "Front", pos: [19, 10, 3.0], hd: -150 },
];
const HALF = D(42),
  RANGE = 11;

export function mountHero(canvas: HTMLCanvasElement, tagLayer: HTMLElement) {
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const mode = "night" as const;
  const P = PALETTE[mode];

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.setClearColor(P.bg);

  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(P.fog[0], P.fog[1], P.fog[2]);
  scene.environment = studioEnvironment(renderer, mode);
  const cam = new THREE.PerspectiveCamera(40, 1, 0.5, 500);
  const M = createMaterials();

  const hemi = new THREE.HemisphereLight(P.hemi[0], P.hemi[1], P.hemi[2]);
  scene.add(hemi);
  const key = new THREE.DirectionalLight(0xffffff, P.key);
  key.position.set(-20, 40, 18);
  scene.add(key);
  const add = <T extends THREE.Object3D>(obj: T, pos: THREE.Vector3): T => {
    obj.position.copy(pos);
    scene.add(obj);
    return obj;
  };

  // ---------- ground, road ----------
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(160, 160), M.ground);
  ground.rotation.x = -Math.PI / 2;
  add(ground, W(2, -6, 0));
  const road = new THREE.Mesh(new THREE.PlaneGeometry(160, 9), M.road);
  road.rotation.x = -Math.PI / 2;
  add(road, W(0, -21, 0.006));
  const dash = new THREE.PlaneGeometry(2.6, 0.14);
  dash.rotateX(-Math.PI / 2);
  for (let x = -60; x <= 60; x += 6) add(new THREE.Mesh(dash, M.lane), W(x, -21, 0.012));
  const edgeG = new THREE.PlaneGeometry(160, 0.1);
  edgeG.rotateX(-Math.PI / 2);
  add(new THREE.Mesh(edgeG, M.lane), W(0, -16.9, 0.012));
  add(new THREE.Mesh(edgeG, M.lane), W(0, -25.1, 0.012));
  const driveG = new THREE.PlaneGeometry(4.6, 9);
  driveG.rotateX(-Math.PI / 2);
  add(new THREE.Mesh(driveG, M.road), W(0.4, -11.5, 0.008));

  // ---------- houses, garden ----------
  add(makeHouse(M, 10, 8, 3.0, 2.0, { windows: true }), W(-7, 8, 0));
  add(makeHouse(M, 6, 5, 2.4, 1.3), W(12, 9.5, 0));
  for (const [x, y, w, h, d] of [
    [4.1, 1.5, 2.2, 0.5, 1.0],
    [7.2, 2.95, 2.4, 0.5, 0.9],
    [-5.3, -1, 1.4, 0.45, 2.0],
  ]) {
    const g = new THREE.Group();
    g.add(softBox(w, h, d, 0.2, M.wall));
    const back = softBox(w, 0.42, 0.28, 0.12, M.wall);
    back.position.set(0, h - 0.05, -d / 2 + 0.14);
    g.add(back);
    blob(g, w + 1.0, d + 1.0, 0.7);
    add(g, W(x, y, 0));
  }
  for (const [x, y] of [
    [3.2, -12.6],
    [3.9, -12.6],
  ]) {
    const g = new THREE.Group();
    g.add(softBox(0.55, 1.0, 0.6, 0.08, M.trim));
    blob(g, 1.1, 1.1, 0.7);
    add(g, W(x, y, 0));
  }
  for (const [x, y, r, h] of [
    [-17, -9, 2.4, 4.2],
    [17, 3, 2.8, 4.8],
    [-16, 13, 2.2, 4],
    [4, 13, 2.0, 3.6],
    [18, -10, 1.8, 3.4],
    [-20, 2, 2.0, 3.8],
  ])
    add(makeTree(M, r, h), W(x, y, 0));

  // fence: slim posts and two rails, with a gap at the side gate
  const postG = new THREE.CylinderGeometry(0.05, 0.05, 1.3, 10);
  const fenceRun = (x0: number, x1: number, y: number) => {
    const railG = new THREE.CylinderGeometry(0.03, 0.03, x1 - x0, 8);
    railG.rotateZ(Math.PI / 2);
    for (const h of [0.45, 1.1]) add(new THREE.Mesh(railG, M.trim), W((x0 + x1) / 2, y, h));
    for (let x = x0; x <= x1 + 0.01; x += 2.5) add(new THREE.Mesh(postG, M.trim), W(x, y, 0.65));
  };
  fenceRun(-22, 6.5, -15.6);
  fenceRun(10, 22, -15.6);

  // ---------- cars ----------
  const parked = add(makeCar(M, M.paintA), W(0.4, -10.8, 0));
  parked.rotation.y = Math.PI / 2;
  const passing = add(makeCar(M, M.paintB), W(-40, -19, 0));
  const passing2 = add(makeCar(M, M.paintA), W(30, -23, 0));
  passing2.rotation.y = Math.PI;

  // ---------- cameras: mount, view fan on the ground, thin frustum ----------
  const anchors: THREE.Vector3[] = [];
  CAMS.forEach((c, i) => {
    const [x, y, h] = c.pos,
      hd = D(c.hd);
    add(makeCameraMount(M, h), W(x, y, 0));
    const selected = i === 1;
    const fan = new THREE.Mesh(
      new THREE.CircleGeometry(RANGE, 48, hd - HALF, HALF * 2),
      new THREE.MeshBasicMaterial({ transparent: true, depthWrite: false, color: selected ? P.sel : P.fan, opacity: selected ? P.selOp : P.fanOp }),
    );
    fan.rotation.x = -Math.PI / 2;
    fan.renderOrder = 2;
    add(fan, W(x, y, 0.03));
    const fwd = new THREE.Vector3(Math.cos(hd), Math.sin(hd), -0.35).normalize();
    const right = new THREE.Vector3().crossVectors(fwd, new THREE.Vector3(0, 0, 1)).normalize();
    const up = new THREE.Vector3().crossVectors(right, fwd);
    const o = new THREE.Vector3(x, y, h),
      L = 4.5;
    const cn = [
      [-1, -1],
      [1, -1],
      [1, 1],
      [-1, 1],
    ].map(([sx, sy]) => o.clone().add(fwd.clone().add(right.clone().multiplyScalar(sx * 0.85)).add(up.clone().multiplyScalar(sy * 0.5)).multiplyScalar(L)));
    const T = (v: THREE.Vector3) => W(v.x, v.y, v.z);
    const pts: THREE.Vector3[] = [];
    cn.forEach((k) => pts.push(T(o), T(k)));
    for (let k = 0; k < 4; k++) pts.push(T(cn[k]), T(cn[(k + 1) % 4]));
    scene.add(
      new THREE.LineSegments(
        new THREE.BufferGeometry().setFromPoints(pts),
        new THREE.LineBasicMaterial({ transparent: true, color: selected ? P.sel : P.line, opacity: selected ? 0.9 : P.lineOp }),
      ),
    );
    anchors.push(W(x, y, h + 0.6));
  });

  // ---------- blind spots: soft rounded patches, named ----------
  const patch = (cx: number, cy: number, w: number, d: number) => {
    const m = softBox(w, 0.01, d, Math.min(w, d) * 0.3, new THREE.MeshBasicMaterial({ transparent: true, depthWrite: false, color: P.blind, opacity: P.blindOp }));
    m.renderOrder = 2;
    add(m, W(cx, cy, 0.02));
  };
  const BLIND = [
    { label: "West lawn · not covered", at: W(-17, -4, 0.4), area: [-17, -4, 8, 8] },
    { label: "Side gate · not covered", at: W(8.3, -13.4, 0.4), area: [8.3, -13.4, 4, 3.6] },
  ];
  BLIND.forEach((b) => patch(b.area[0], b.area[1], b.area[2], b.area[3]));

  // ---------- person with a pool of light ----------
  const person = makePerson(M);
  person.add(makePool(mode));
  const personLight = new THREE.PointLight(0xffffff, P.pLight, 8, 2);
  personLight.position.y = 2.4;
  person.add(personLight);
  scene.add(person);
  const route = [
    [2, -14],
    [4, -8],
    [6.5, -3.5],
    [3, 0],
    [-2, -2],
    [-4, -8],
    [-1, -13.5],
  ];
  const curve = new THREE.CatmullRomCurve3(
    route.map(([x, y]) => W(x, y, 0)),
    true,
    "centripetal",
  );

  applyMode(M, mode);

  // ---------- tags: pills for cameras and blind spots, a card for the person ----------
  const pill = (text: string, dot = "") => {
    const el = document.createElement("div");
    el.className = "pill hero-tag";
    el.innerHTML = `<span class="dot ${dot}"></span><span></span>`;
    (el.lastChild as HTMLElement).textContent = text;
    tagLayer.appendChild(el);
    return el;
  };
  const tags: { el: HTMLElement; at: () => THREE.Vector3 }[] = [];
  CAMS.forEach((c, i) => tags.push({ el: pill(c.name, i === 1 ? "clear" : ""), at: () => anchors[i] }));
  BLIND.forEach((b) => tags.push({ el: pill(b.label, "alert"), at: () => b.at }));
  const card = document.createElement("div");
  card.className = "hero-card hero-tag";
  card.innerHTML = "<strong>Person</strong><span></span>";
  tagLayer.appendChild(card);
  const cardMeta = card.lastChild as HTMLElement;
  tags.push({ el: card, at: () => person.position.clone().add(new THREE.Vector3(0, 2.6, 0)) });

  // ---------- view ----------
  const target = W(1, -4, 0);
  let theta = D(-38);
  const phi = D(52);
  let radius = 54;
  const v = new THREE.Vector3();
  function resize() {
    const w = canvas.clientWidth,
      h = canvas.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    cam.aspect = w / h;
    cam.fov = w < 760 ? 52 : 40;
    radius = w < 760 ? 64 : 54;
    // Lift the scene into the top of the hero; the copy sits over the lower part.
    cam.setViewOffset(w, h, 0, h * (w < 760 ? 0.18 : 0.12), w, h);
    cam.updateProjectionMatrix();
  }
  function place(el: HTMLElement, p: THREE.Vector3) {
    v.copy(p).project(cam);
    const w = canvas.clientWidth,
      h = canvas.clientHeight;
    const sx = (v.x * 0.5 + 0.5) * w,
      sy = (-v.y * 0.5 + 0.5) * h;
    // Off screen, or drifting into the headline (lower left of the hero): hide.
    if (v.z > 1 || Math.abs(v.x) > 1.05 || Math.abs(v.y) > 1.05 || (sy > h * 0.46 && sx < Math.min(w * 0.62, 980))) {
      el.style.visibility = "hidden";
      return;
    }
    el.style.visibility = "visible";
    el.style.transform = `translate(${sx}px, ${sy}px) translate(-50%, -100%)`;
  }

  let t = 0.18,
    last = performance.now(),
    visible = true,
    raf = 0;
  function frame(now: number) {
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    if (!reduce) {
      theta += dt * 0.035;
      t = (t + dt * 0.012) % 1;
      passing.position.x += dt * 9;
      if (passing.position.x > 50) passing.position.x = -50;
      passing2.position.x -= dt * 7;
      if (passing2.position.x < -50) passing2.position.x = 50;
    }
    cam.position.set(target.x + radius * Math.sin(phi) * Math.sin(theta), target.y + radius * Math.cos(phi), target.z + radius * Math.sin(phi) * Math.cos(theta));
    cam.lookAt(target);
    const p = curve.getPointAt(t),
      ahead = curve.getPointAt((t + 0.01) % 1);
    person.position.copy(p);
    person.lookAt(ahead.x, 0, ahead.z);
    const wx = p.x,
      wy = -p.z;
    let seen = 0;
    for (const c of CAMS) {
      const dx = wx - c.pos[0],
        dy = wy - c.pos[1];
      let a = Math.atan2(dy, dx) - D(c.hd);
      a = Math.atan2(Math.sin(a), Math.cos(a));
      if (Math.abs(a) < HALF && Math.hypot(dx, dy) < RANGE) seen++;
    }
    cardMeta.textContent = `${Math.hypot(wx + 5.6, wy - 4).toFixed(0)} m from the door · ${
      seen ? `seen by ${seen} camera${seen > 1 ? "s" : ""}` : "not covered here"
    }`;
    for (const tg of tags) place(tg.el, tg.at());
    renderer.render(scene, cam);
    if (!reduce && visible) raf = requestAnimationFrame(frame);
  }

  new IntersectionObserver(([e]) => {
    visible = e.isIntersecting;
    if (visible && !reduce) {
      last = performance.now();
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(frame);
    }
  }).observe(canvas);
  window.addEventListener("resize", () => {
    resize();
    if (reduce) frame(performance.now());
  });
  resize();
  frame(performance.now());
}
