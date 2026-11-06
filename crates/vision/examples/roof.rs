//! Outline roofs in the site photo from clicks. Diagnostic.
//! cargo run --release -p minastir-vision --example roof -- <photo> <scale m/px> <out.png> u,v [u,v …]
use minastir_vision::roof;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let img = roof::remove_labels(&image::open(&a[0])?.to_rgb8());
    img.save(std::path::Path::new(&a[2]).with_extension("clean.png"))?;
    let scale: f64 = a[1].parse()?;
    let mut out = img.clone();
    let mut sam = std::env::var("SAM_MODELS").ok().map(|d| roof::Sam::load(std::path::Path::new(&d)).expect("sam"));
    for (k, c) in a[3..].iter().enumerate() {
        // "u,v" or "u,v+u2,v2-u3,v3": extra include (+) / exclude (−) points.
        let mut prompts = Vec::new();
        let mut rest = c.as_str();
        let mut include = true;
        loop {
            let end = rest.find(['+', '-']).unwrap_or(rest.len());
            let (u, v) = rest[..end].split_once(',').map(|(u, v)| (u.parse::<f64>().unwrap(), v.parse::<f64>().unwrap())).unwrap();
            prompts.push(roof::Prompt { u, v, include });
            if end == rest.len() {
                break;
            }
            include = &rest[end..end + 1] == "+";
            rest = &rest[end + 1..];
        }
        let (u, v) = (prompts[0].u, prompts[0].v);
        let t = std::time::Instant::now();
        let found = match sam.as_mut() {
            Some(s) => s.outline_prompts(&img, 1, &prompts, scale).expect("sam run"),
            None => roof::outline(&img, (u, v), scale),
        };
        match found {
            Some(poly) => {
                let n = poly.len();
                let area = (0..n).map(|i| poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1]).sum::<f64>().abs() / 2.0 * scale * scale;
                println!("click {k} ({u},{v}): {n} corners, {area:.0} m² in {:?}", t.elapsed());
                for i in 0..n {
                    let (p, q) = (poly[i], poly[(i + 1) % n]);
                    let steps = ((q[0] - p[0]).hypot(q[1] - p[1]) * 2.0) as usize + 1;
                    for s in 0..=steps {
                        let f = s as f64 / steps as f64;
                        let (x, y) = (p[0] + (q[0] - p[0]) * f, p[1] + (q[1] - p[1]) * f);
                        for d in -1i64..=1 {
                            let (xx, yy) = (x as i64 + d, y as i64);
                            if xx >= 0 && yy >= 0 && (xx as u32) < out.width() && (yy as u32) < out.height() {
                                out.put_pixel(xx as u32, yy as u32, image::Rgb([255, 30, 30]));
                            }
                        }
                    }
                }
            }
            None => println!("click {k} ({u},{v}): no roof"),
        }
    }
    out.save(&a[2])?;
    Ok(())
}
