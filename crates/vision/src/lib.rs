//! Live scene understanding: detect objects in camera frames, place them on the ground using the
//! camera layout, and track them across cameras. Plus one-off calibration from background images.

pub mod align;
pub mod appearance;
pub mod assoc;
pub mod calibrate;
pub mod describe;
pub mod detector;
pub mod fit;
pub mod geometry;
pub mod groundmap;
pub mod imgtrack;
pub mod live;
pub mod matcher;
pub mod moge;
pub mod pose;
pub mod onnx;
pub mod register;
pub mod resect;
pub mod roof;
pub mod scenemap;
pub mod segment;
pub mod shapes;
pub mod sitealign;
pub mod street;
pub mod survey;
pub mod tracker;
pub mod distortion;
pub mod wheels;
