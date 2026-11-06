//! Print an ONNX model's inputs/outputs: cargo run --release -p minastir-vision --example inspect -- model.onnx
fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).expect("model path");
    let env = ort::init().build().map_err(|e| anyhow::anyhow!("{e}"))?;
    let s = ort::session::Session::builder(&env).map_err(|e| anyhow::anyhow!("{e}"))?
        .commit_from_file(&path).map_err(|e| anyhow::anyhow!("{e}"))?;
    for i in s.inputs() { println!("in  {} {:?}", i.name(), i.dtype()); }
    for o in s.outputs() { println!("out {} {:?}", o.name(), o.dtype()); }
    Ok(())
}
