//! Render and compare: find where a vehicle stands and which way it points by drawing a 3D car
//! into every camera that sees it and matching the drawings to what the cameras actually show.
//!
//! The car is a few boxes (body, cabin; a ute's cab sits at the front) with a position, heading
//! and class-typical size. Each camera's view of it is a silhouette — the outline Segment Anything
//! cut from that camera's image. A candidate pose is drawn through each camera's calibrated lens
//! (pinhole or panorama) and scored by how well the drawn outline overlaps the real one; the
//! best pose across all cameras wins. Nothing here says which way cars "usually" face: the pixels
//! decide.
//!
//! A plain car outline looks the same nose-first or tail-first, so for cars the result is the
//! car's axis (heading mod 180°); which end is the front needs appearance (learned separately).

use crate::{geometry::CameraModel, roof::BoxMask};

/// One box of a vehicle shape, in the vehicle's frame: along its length from `from` to `to`
/// (fractions of the length, −0.5 = tail, +0.5 = nose), `width` (fraction of the vehicle's
/// width), between heights `h0` and `h1` (metres).
#[derive(Debug, Clone, Copy)]
pub struct Part {
    pub from: f64,
    pub to: f64,
    pub width: f64,
    pub h0: f64,
    pub h1: f64,
}

#[derive(Debug, Clone)]
pub struct Shape {
    pub length: f64,
    pub width: f64,
    pub parts: Vec<Part>,
    /// Looks the same turned 180° (then only the axis can be found from outlines).
    pub symmetric: bool,
}

/// Typical shape for a detector class.
pub fn shape_for(class: &str) -> Shape {
    let p = |from, to, width, h0, h1| Part { from, to, width, h0, h1 };
    match class {
        // Utes and vans: tall cab at the front, low tray behind.
        "truck" => Shape {
            length: 5.3,
            width: 1.9,
            parts: vec![p(-0.5, 0.5, 1.0, 0.0, 1.0), p(-0.02, 0.42, 0.95, 1.0, 1.85), p(-0.5, -0.02, 1.0, 1.0, 1.2)],
            symmetric: false,
        },
        "bus" => Shape { length: 10.0, width: 2.5, parts: vec![p(-0.5, 0.5, 1.0, 0.0, 3.0)], symmetric: true },
        _ => Shape {
            length: 4.5,
            width: 1.8,
            parts: vec![p(-0.5, 0.5, 1.0, 0.0, 0.85), p(-0.25, 0.25, 0.88, 0.85, 1.45)],
            symmetric: true,
        },
    }
}

/// A vehicle pose on the ground: centre (x, z) and heading (degrees, camera-yaw convention:
/// direction (sin θ, cos θ)).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pose2 {
    pub x: f64,
    pub z: f64,
    pub heading: f64,
}

/// What one camera shows of the vehicle: its outline, on a coarse grid over a window of the
/// image, plus the camera model and image size to draw into it.
pub struct View<'a> {
    pub model: &'a CameraModel,
    /// Image size in pixels (the mask's coordinates).
    pub image: (f64, f64),
    /// Height of the ground under the vehicle in this camera's frame (the ground isn't flat:
    /// a car 30 m away may stand a metre above or below the camera's own ground plane).
    pub base: f64,
    grid: Grid,
    target: Vec<bool>,
}

#[derive(Debug, Clone, Copy)]
struct Grid {
    x0: f64,
    y0: f64,
    cell: f64,
    w: usize,
    h: usize,
}

/// Grid cells along the window's long side.
const CELLS: f64 = 80.0;

impl<'a> View<'a> {
    pub fn new(model: &'a CameraModel, image: (f64, f64), mask: &BoxMask) -> Self {
        let cell = mask.width.max(mask.height) as f64 / CELLS;
        let (w, h) = (((mask.width as f64) / cell).ceil() as usize, ((mask.height as f64) / cell).ceil() as usize);
        let grid = Grid { x0: mask.x0, y0: mask.y0, cell, w, h };
        let mut target = vec![false; w * h];
        for gy in 0..h {
            for gx in 0..w {
                let (mx, my) = ((((gx as f64) + 0.5) * cell) as usize, (((gy as f64) + 0.5) * cell) as usize);
                if mx < mask.width && my < mask.height {
                    target[gy * w + gx] = mask.cells[my * mask.width + mx];
                }
            }
        }
        Self { model, image, base: 0.0, grid, target }
    }

    /// Stand the vehicle on ground `base` metres above this camera's ground plane.
    pub fn on_ground(mut self, base: f64) -> Self {
        self.base = base;
        self
    }

    /// The vehicle's outline as polygons in image pixels (one convex polygon per part), or None
    /// if any part is behind the camera or far outside its view (just past the frame edge is
    /// fine: the comparison only counts what's inside the image).
    pub fn outline(&self, pose: Pose2, shape: &Shape) -> Option<Vec<Vec<[f64; 2]>>> {
        let (s, c) = pose.heading.to_radians().sin_cos();
        let (fwd, right) = ([s, c], [c, -s]);
        let mut polys = Vec::with_capacity(shape.parts.len());
        for p in &shape.parts {
            let mut pts = Vec::with_capacity(8);
            for along in [p.from, p.to] {
                for side in [-0.5, 0.5] {
                    let (a, b) = (along * shape.length, side * p.width * shape.width);
                    let (x, z) = (pose.x + fwd[0] * a + right[0] * b, pose.z + fwd[1] * a + right[1] * b);
                    for h in [p.h0, p.h1] {
                        let (u, v) = self.model.project_beyond(x, z, h + self.base)?;
                        pts.push([u * self.image.0, v * self.image.1]);
                    }
                }
            }
            polys.push(convex_hull(pts));
        }
        Some(polys)
    }

    /// Overlap of the drawn outline with the real one (intersection over union), 0 if the pose
    /// can't be drawn.
    pub fn score(&self, pose: Pose2, shape: &Shape) -> f64 {
        let Some(polys) = self.outline(pose, shape) else { return 0.0 };
        let g = self.grid;
        let mut drawn = vec![false; g.w * g.h];
        for poly in &polys {
            // Cells whose centres fall inside the polygon (within its bounding box).
            let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
            for p in poly {
                (x0, y0, x1, y1) = (x0.min(p[0]), y0.min(p[1]), x1.max(p[0]), y1.max(p[1]));
            }
            let gx0 = (((x0 - g.x0) / g.cell).floor().max(0.0)) as usize;
            let gy0 = (((y0 - g.y0) / g.cell).floor().max(0.0)) as usize;
            let gx1 = (((x1 - g.x0) / g.cell).ceil().max(0.0) as usize).min(g.w);
            let gy1 = (((y1 - g.y0) / g.cell).ceil().max(0.0) as usize).min(g.h);
            for gy in gy0..gy1 {
                for gx in gx0..gx1 {
                    let pt = [g.x0 + (gx as f64 + 0.5) * g.cell, g.y0 + (gy as f64 + 0.5) * g.cell];
                    if inside_convex(poly, pt) {
                        drawn[gy * g.w + gx] = true;
                    }
                }
            }
        }
        let (mut inter, mut union) = (0usize, 0usize);
        for (d, t) in drawn.iter().zip(&self.target) {
            inter += (*d && *t) as usize;
            union += (*d || *t) as usize;
        }
        if union == 0 {
            0.0
        } else {
            inter as f64 / union as f64
        }
    }

    /// Share of the window the real outline covers (for diagnostics).
    pub fn coverage(&self) -> f64 {
        self.target.iter().filter(|t| **t).count() as f64 / self.target.len().max(1) as f64
    }
}

#[derive(Debug, Clone)]
pub struct Fit {
    pub pose: Pose2,
    /// Mean overlap across views (0–1), and each view's.
    pub score: f64,
    pub per_view: Vec<f64>,
    /// Best score with the vehicle turned 90° — how decisive the heading is.
    pub across: f64,
    /// Heading is only known up to 180° (symmetric outline).
    pub axis_only: bool,
}

/// Search for the pose that best explains every view, within 4 m of `start`.
pub fn fit(views: &[View], shape: &Shape, start: (f64, f64)) -> Fit {
    fit_within(views, shape, start, 4.0)
}

/// Search for the pose that best explains every view, within `reach` metres of `start`.
pub fn fit_within(views: &[View], shape: &Shape, start: (f64, f64), reach: f64) -> Fit {
    fit_guided(views, shape, start, reach, None)
}

/// `fit_within`, with the heading kept within `tol` degrees of a known axis (`axis`, modulo
/// 180°) when one is given: the outlines then only place the vehicle and settle its exact angle.
pub fn fit_guided(views: &[View], shape: &Shape, start: (f64, f64), reach: f64, hint: Option<(f64, f64)>) -> Fit {
    let allowed = |h: f64| hint.is_none_or(|(axis, tol)| angle_diff(h, axis, 180.0) <= tol);
    let total = |p: Pose2| views.iter().map(|v| v.score(p, shape)).sum::<f64>() / views.len().max(1) as f64;
    let span = if shape.symmetric { 180.0 } else { 360.0 };
    // Coarse: headings every 6°, centres on a 0.5 m grid (finer when the reach is small).
    let step = (reach / 8.0).clamp(0.15, 0.5);
    let n = (reach / step).round() as i32;
    let mut best = (f64::MIN, Pose2 { x: start.0, z: start.1, heading: 0.0 });
    let mut by_heading: Vec<(f64, f64)> = Vec::new(); // (heading, best score at it)
    // With an axis hint, start the sweep on the axis itself so it's always tried.
    let mut h = hint.map_or(0.0, |(axis, tol)| (axis - tol).rem_euclid(span));
    let sweep_end = h + span;
    while h < sweep_end {
        if !allowed(h) {
            h += 6.0;
            continue;
        }
        let mut at = f64::MIN;
        for i in -n..=n {
            for j in -n..=n {
                let p = Pose2 { x: start.0 + i as f64 * step, z: start.1 + j as f64 * step, heading: h.rem_euclid(span) };
                let s = total(p);
                at = at.max(s);
                if s > best.0 {
                    best = (s, p);
                }
            }
        }
        by_heading.push((h.rem_euclid(span), at));
        h += 6.0;
    }
    // Fine: shrink the steps around the best.
    let (mut step_xz, mut step_h) = (step / 2.0, 3.0);
    for _ in 0..4 {
        let c = best.1;
        for dh in -2..=2 {
            for i in -2..=2 {
                for j in -2..=2 {
                    let p = Pose2 { x: c.x + i as f64 * step_xz, z: c.z + j as f64 * step_xz, heading: (c.heading + dh as f64 * step_h).rem_euclid(span) };
                    if !allowed(p.heading) {
                        continue;
                    }
                    let s = total(p);
                    if s > best.0 {
                        best = (s, p);
                    }
                }
            }
        }
        step_xz /= 2.0;
        step_h /= 2.0;
    }
    let ortho = (best.1.heading + 90.0).rem_euclid(span);
    let across = by_heading
        .iter()
        .filter(|(h, _)| angle_diff(*h, ortho, span) <= 15.0)
        .map(|(_, s)| *s)
        .fold(0.0, f64::max);
    let pose = best.1;
    Fit { pose, score: best.0, per_view: views.iter().map(|v| v.score(pose, shape)).collect(), across, axis_only: shape.symmetric }
}

/// Diagnostic: for each heading (every `every` degrees), the best overlap and where, searching
/// centres within `reach` of `start`.
pub fn heading_profile(views: &[View], shape: &Shape, start: (f64, f64), reach: f64, every: f64) -> Vec<(f64, f64, Pose2)> {
    let total = |p: Pose2| views.iter().map(|v| v.score(p, shape)).sum::<f64>() / views.len().max(1) as f64;
    let step = 0.25;
    let n = (reach / step).round() as i32;
    let mut out = Vec::new();
    let mut h = 0.0;
    while h < 360.0 {
        let mut best = (f64::MIN, Pose2 { x: start.0, z: start.1, heading: h });
        for i in -n..=n {
            for j in -n..=n {
                let p = Pose2 { x: start.0 + i as f64 * step, z: start.1 + j as f64 * step, heading: h };
                let s = total(p);
                if s > best.0 {
                    best = (s, p);
                }
            }
        }
        out.push((h, best.0, best.1));
        h += every;
    }
    out
}

fn angle_diff(a: f64, b: f64, span: f64) -> f64 {
    let d = (a - b).rem_euclid(span);
    d.min(span - d)
}

fn convex_hull(mut pts: Vec<[f64; 2]>) -> Vec<[f64; 2]> {
    pts.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    let cross = |o: [f64; 2], a: [f64; 2], b: [f64; 2]| (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    let mut hull: Vec<[f64; 2]> = Vec::with_capacity(pts.len() * 2);
    for pass in 0..2 {
        let start = hull.len();
        let iter: Box<dyn Iterator<Item = &[f64; 2]>> = if pass == 0 { Box::new(pts.iter()) } else { Box::new(pts.iter().rev()) };
        for &p in iter {
            while hull.len() >= start + 2 && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
                hull.pop();
            }
            hull.push(p);
        }
        hull.pop();
    }
    hull
}

/// Point in a convex polygon given in counter-clockwise order (as `convex_hull` returns).
fn inside_convex(poly: &[[f64; 2]], p: [f64; 2]) -> bool {
    if poly.len() < 3 {
        return false;
    }
    (0..poly.len()).all(|i| {
        let (a, b) = (poly[i], poly[(i + 1) % poly.len()]);
        (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0.0
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use minastir_core::{
        calib::{CameraCalibration, LensCalibration},
        onvif::Pose,
    };

    fn camera(x: f64, z: f64, yaw: f64) -> CameraModel {
        let lens = LensCalibration {
            u0: 0.0, u1: 1.0, hfov: 90.0, vfov: 60.0, pitch: 30.0, roll: 0.0, yaw_offset: 0.0, height: 3.0,
            ground_inliers: 0, ground_rms: 0.0, up: [0.0, -1.0, 0.0], wall_points: 0, ground_slope: 0.0, cylindrical: false, k1: 0.0,
        };
        let calib = CameraCalibration { lenses: vec![lens], pitch: 30.0, roll: 0.0, height: 3.0, computed_at: 0.0, scale: 1.0, alignment: None, fov_correction: 1.0 };
        CameraModel::new(Pose { x, z, height: 3.0, yaw, pitch: 30.0 }, &calib, None)
    }

    /// The real outline of a vehicle at `truth`, as a camera would show it.
    fn mask_of(model: &CameraModel, truth: Pose2, shape: &Shape) -> BoxMask {
        let image = (1920.0, 1080.0);
        let probe = View { model, image, base: 0.0, grid: Grid { x0: 0.0, y0: 0.0, cell: 1.0, w: 1, h: 1 }, target: vec![false] };
        let polys = probe.outline(truth, shape).expect("visible");
        let (w, h) = (1920usize, 1080usize);
        let mut cells = vec![false; w * h];
        for y in (0..h).step_by(2) {
            for x in (0..w).step_by(2) {
                let p = [x as f64 + 1.0, y as f64 + 1.0];
                if polys.iter().any(|poly| inside_convex(poly, p)) {
                    for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        cells[(y + dy) * w + x + dx] = true;
                    }
                }
            }
        }
        BoxMask { x0: 0.0, y0: 0.0, width: w, height: h, cells }
    }

    #[test]
    fn finds_a_parked_cars_axis_from_two_cameras() {
        let shape = shape_for("car");
        let truth = Pose2 { x: 1.0, z: 9.0, heading: 62.0 };
        let (a, b) = (camera(0.0, 0.0, 0.0), camera(9.0, 6.0, 280.0));
        let (ma, mb) = (mask_of(&a, truth, &shape), mask_of(&b, truth, &shape));
        let views = [View::new(&a, (1920.0, 1080.0), &ma), View::new(&b, (1920.0, 1080.0), &mb)];
        let f = fit(&views, &shape, (1.8, 8.2)); // start 1.1 m off
        assert!(angle_diff(f.pose.heading, 62.0, 180.0) < 4.0, "{f:?}");
        assert!((f.pose.x - 1.0).hypot(f.pose.z - 9.0) < 0.3, "{f:?}");
        assert!(f.score > 0.85 && f.across < f.score - 0.2, "{f:?}");
    }

    #[test]
    fn a_utes_cab_tells_front_from_back() {
        let shape = shape_for("truck");
        let truth = Pose2 { x: -2.0, z: 10.0, heading: 200.0 };
        let a = camera(0.0, 0.0, 0.0);
        let ma = mask_of(&a, truth, &shape);
        let f = fit(&[View::new(&a, (1920.0, 1080.0), &ma)], &shape, (-2.0, 10.0));
        assert!(angle_diff(f.pose.heading, 200.0, 360.0) < 8.0, "{f:?}");
    }
}
