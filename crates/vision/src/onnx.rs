//! Shared ONNX Runtime setup: one process-wide environment, and sessions on the Apple GPU
//! (CoreML, MLProgram format) or the CPU.

use anyhow::{anyhow, Context, Result};
use ort::{
    ep,
    session::{builder::GraphOptimizationLevel, Session},
};
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    Cpu,
    /// Metal GPU through CoreML. Needs fully static input shapes (see `dims`).
    Gpu,
}

/// ONNX Runtime allows one environment per process; every session shares it.
fn environment() -> Result<&'static ort::environment::Environment> {
    static ENV: std::sync::OnceLock<ort::environment::Environment> = std::sync::OnceLock::new();
    if let Some(e) = ENV.get() {
        return Ok(e);
    }
    let env = ort::init().with_name("minastir").build().map_err(|e| anyhow!("{e}"))?;
    Ok(ENV.get_or_init(|| env))
}

/// Build a session. `dims` pins named symbolic input dimensions (e.g. batch_size=1), which lets ORT
/// constant-fold shape arithmetic — CoreML rejects unbounded dimensions and falls back to CPU.
pub fn session(model: &Path, device: Device, dims: &[(&str, i64)]) -> Result<Session> {
    let mut b = Session::builder(environment()?)
        .map_err(|e| anyhow!("{e}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow!("{e}"))?;
    for (name, size) in dims {
        b = b.with_dimension_override(*name, *size).map_err(|e| anyhow!("{e}"))?;
    }
    // Always name the providers explicitly.
    let cpu = ep::CPU::default().build();
    b = match device {
        Device::Cpu => b.with_execution_providers([cpu]),
        Device::Gpu => {
            let cache = model.with_extension("coreml-cache");
            std::fs::create_dir_all(&cache).ok();
            // Measured on an M5 Max with D-FINE-M: MLProgram on the GPU ≈ 12 ms/frame vs ≈ 71 ms on
            // CPU. The older NeuralNetwork format mis-infers shapes on transformer decoders, and
            // "All" adds a multi-second Neural Engine compile for no gain.
            let units = match std::env::var("MINASTIR_COREML_UNITS").as_deref() {
                Ok("all") => ep::coreml::ComputeUnits::All,
                Ok("ane") => ep::coreml::ComputeUnits::CPUAndNeuralEngine,
                _ => ep::coreml::ComputeUnits::CPUAndGPU,
            };
            let coreml = ep::CoreML::default()
                .with_model_format(ep::coreml::ModelFormat::MLProgram)
                .with_compute_units(units)
                .with_static_input_shapes(true)
                .with_model_cache_dir(cache.display())
                .build()
                .error_on_failure();
            b.with_execution_providers([coreml, cpu])
        }
    }
    .map_err(|e| anyhow!("{e}"))?;
    b.commit_from_file(model).map_err(|e| anyhow!("{e}")).with_context(|| format!("loading {}", model.display()))
}
