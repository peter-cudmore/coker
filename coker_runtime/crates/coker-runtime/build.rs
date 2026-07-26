fn main() {
    println!("cargo:rustc-check-cfg=cfg(osqp_embedded)");
    println!("cargo:rustc-cfg=osqp_embedded");
}
