use std::env;

fn main() {
    println!("cargo:rerun-if-env-changed=CARGO_CFG_TARGET_OS");
    println!("cargo:rustc-check-cfg=cfg(osqp_embedded)");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("none") {
        println!("cargo:rustc-cfg=osqp_embedded");
    }
}
