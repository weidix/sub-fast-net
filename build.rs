fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    let profile = std::env::var("PROFILE").unwrap_or_default();

    if target_os == "windows" && target_env == "msvc" && profile == "debug" {
        // Evidence from 2026-06-18 investigation:
        // the default MSVC debug executable had a 1 MiB stack reserve
        // (`dumpbin /headers target\debug\sub-fast-net.exe`), and CUDA debug
        // training with batch_size=1 reached `after_backward` then overflowed
        // before `optimizer.step(...)` returned. The same finite path completed
        // both the tiny repro and `configs/profile_current.toml` when linked
        // with `/STACK:8388608`, while release CUDA training already completed.
        // This preserves training semantics and does not alter CUDA event
        // profiling; it only gives the debug main thread enough stack for Burn
        // 0.21's CUDA autodiff/Adam optimizer path.
        println!("cargo:rustc-link-arg=/STACK:8388608");
    }
}
