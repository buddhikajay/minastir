//! Print an ONNX model's inputs and outputs. Diagnostic.
use minastir_vision::onnx::{self, Device};
fn main() -> anyhow::Result<()> {
    for p in std::env::args().skip(1) {
        let s = onnx::session(std::path::Path::new(&p), Device::Cpu, &[])?;
        println!("{p}");
        for i in s.inputs() { println!("  in  {} {:?}", i.name(), i.dtype()); }
        for o in s.outputs() { println!("  out {} {:?}", o.name(), o.dtype()); }
    }
    Ok(())
}
