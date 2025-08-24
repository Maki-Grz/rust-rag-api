use twox_hash::XxHash3_64;

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn compute_hash(s: &str) -> u64 {
    let hash = XxHash3_64::oneshot(s.as_bytes());
    hash
}
