#[path = "src/generated_solver_contract.rs"]
mod generated_solver_contract;

extern crate cmake;
use cmake::Config;

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!(
        "cargo:rerun-if-env-changed={}",
        generated_solver_contract::GENERATED_SOLVER_ENV
    );
    println!("cargo:rerun-if-env-changed=CARGO_CFG_TARGET_OS");
    println!("cargo:rerun-if-env-changed=COKER_OSQP_CMAKE_TOOLCHAIN_FILE");
    println!("cargo:rerun-if-changed=cmake/thumbv7em-none-eabihf.cmake");
    println!("cargo:rerun-if-changed=embedded_bindings_wrapper.h");

    println!("cargo:rustc-check-cfg=cfg(osqp_dlong)");
    println!("cargo:rustc-check-cfg=cfg(osqp_embedded)");
    println!("cargo:rustc-check-cfg=cfg(osqp_generated_solver)");
    println!("cargo:rustc-cfg=osqp_embedded");

    if !Path::new("osqp/README.md").exists() {
        let _ = Command::new("git")
            .args(["submodule", "update", "--init", "--recursive"])
            .status();
    }

    let generated_solver_dir =
        env::var_os(generated_solver_contract::GENERATED_SOLVER_ENV).map(PathBuf::from);

    if generated_solver_dir.is_some() && env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("none") {
        panic!(
            "{} requires target_os=none because the supplied OSQP 0.6.3 generated artifacts are built against the embedded runtime.",
            generated_solver_contract::GENERATED_SOLVER_ENV
        );
    }
    if let Some(generated_solver_dir) = generated_solver_dir.as_ref() {
        validate_generated_solver_layout(generated_solver_dir);
        println!("cargo:rustc-cfg=osqp_generated_solver");
    }

    let out_dir = env::var("OUT_DIR").unwrap();
    let src_dir = Path::new(&out_dir).join("src");
    let build_dir = Path::new(&out_dir).join("build");
    let _ = fs::remove_dir_all(&src_dir);
    fs::create_dir_all(&src_dir).expect("failed to create OSQP sources directory in `OUT_DIR`");

    fs_extra::dir::copy(
        "osqp",
        &src_dir,
        &fs_extra::dir::CopyOptions {
            overwrite: true,
            skip_exist: false,
            content_only: true,
            ..fs_extra::dir::CopyOptions::new()
        },
    )
    .expect("failed to copy OSQP sources to `OUT_DIR`");

    let _ = fs::remove_dir_all(&build_dir);
    fs::create_dir_all(&build_dir).expect("failed to create OSQP build directory in `OUT_DIR`");
    let mut config = Config::new(&src_dir);
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("none") {
        let toolchain = env::var_os("COKER_OSQP_CMAKE_TOOLCHAIN_FILE")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("cmake/thumbv7em-none-eabihf.cmake"));
        let toolchain = toolchain.canonicalize().unwrap_or_else(|error| {
            panic!(
                "failed to resolve embedded OSQP CMake toolchain {}: {error}",
                toolchain.display()
            )
        });
        config.generator("Ninja");
        config.define("CMAKE_TOOLCHAIN_FILE", toolchain);
    }
    config
        .define("OSQP_BUILD_UNITTESTS", "OFF")
        .define("OSQP_ALGEBRA_BACKEND", "builtin");
    configure_embedded_osqp(&mut config);
    config.build_target("osqpstatic").build();

    compile_embedded_bridge(&src_dir, &build_dir);
    generate_embedded_bindings(&src_dir, &build_dir, Path::new(&out_dir));
    if let Some(generated_solver_dir) = generated_solver_dir.as_ref() {
        compile_generated_solver(generated_solver_dir, &src_dir);
    }

    let native_lib_dir = if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        let configuration = match env::var("PROFILE").as_deref() {
            Ok("release") => "Release",
            _ => "Debug",
        };
        build_dir.join("out").join(configuration)
    } else {
        build_dir.join("out")
    };
    println!(
        "cargo:rustc-link-search=native={}",
        native_lib_dir.display()
    );
    println!("cargo:rustc-link-lib=static=osqpstatic");
}

fn configure_embedded_osqp(config: &mut Config) {
    config
        .define("OSQP_EMBEDDED_MODE", "2")
        .define("OSQP_USE_FLOAT", "ON")
        .define("OSQP_USE_LONG", "OFF")
        .define("OSQP_ENABLE_PRINTING", "OFF")
        .define("OSQP_ENABLE_PROFILING", "OFF")
        .define("OSQP_ENABLE_INTERRUPT", "OFF")
        .define("OSQP_ENABLE_DERIVATIVES", "OFF");
}

fn compile_embedded_bridge(src_dir: &Path, build_dir: &Path) {
    cc::Build::new()
        .define("EMBEDDED", "2")
        .file("embedded_bridge.c")
        .include(src_dir)
        .include(src_dir.join("include"))
        .include(src_dir.join("include/public"))
        .include(src_dir.join("include/private"))
        .include(src_dir.join("algebra/_common"))
        .include(src_dir.join("algebra/_common/lin_sys/qdldl"))
        .include(build_dir)
        .include(build_dir.join("include"))
        .include(build_dir.join("include/public"))
        .include(build_dir.join("include/private"))
        .include(build_dir.join("_deps/qdldl-build/include"))
        .compile("osqp_embedded_bridge");
}

fn generate_embedded_bindings(src_dir: &Path, build_dir: &Path, out_dir: &Path) {
    let builder = bindgen::Builder::default()
        .header("embedded_bindings_wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .use_core()
        .ctypes_prefix("core::ffi")
        .size_t_is_usize(true)
        .generate_comments(false)
        .layout_tests(false)
        .allowlist_type("OSQP.*")
        .allowlist_type("LinSysSolver")
        .allowlist_type("linsys_solver")
        .allowlist_type("qdldl.*")
        .allowlist_type("OSQPCscMatrix")
        .allowlist_type("OSQPMatrix")
        .allowlist_type("OSQPVectorf")
        .allowlist_type("OSQPVectori")
        .allowlist_type("csc")
        .allowlist_var("OSQP_.*")
        .allowlist_var("QDLDL_.*")
        .allowlist_function("osqp_.*")
        .allowlist_function("init_linsys_solver_qdldl")
        .allowlist_function("solve_linsys_qdldl")
        .allowlist_function("update_settings_linsys_solver_qdldl")
        .allowlist_function("warm_start_linsys_solver_qdldl")
        .allowlist_function("update_linsys_solver_.*")
        .allowlist_function("free_linsys_solver_qdldl")
        .allowlist_function("name_qdldl")
        .clang_arg("-DEMBEDDED=2")
        .clang_arg(format!("-I{}", src_dir.display()))
        .clang_arg(format!("-I{}", src_dir.join("include").display()))
        .clang_arg(format!("-I{}", src_dir.join("include/public").display()))
        .clang_arg(format!("-I{}", src_dir.join("include/private").display()))
        .clang_arg(format!("-I{}", src_dir.join("algebra/_common").display()))
        .clang_arg(format!(
            "-I{}",
            src_dir.join("algebra/_common/lin_sys/qdldl").display()
        ))
        .clang_arg(format!("-I{}", src_dir.join("algebra/builtin").display()))
        .clang_arg(format!("-I{}", build_dir.display()))
        .clang_arg(format!("-I{}", build_dir.join("include").display()))
        .blocklist_function("osqp_adjoint_.*")
        .blocklist_function("osqp_codegen")
        .blocklist_function("osqp_set_default_codegen_defines")
        .clang_arg(format!("-I{}", build_dir.join("include/public").display()))
        .allowlist_function("set_rho_vec")
        .allowlist_function("update_rho_vec")
        .allowlist_function("reset_info")
        .allowlist_function("update_KKT_param2")
        .clang_arg(format!("-I{}", build_dir.join("include/private").display()))
        .clang_arg(format!(
            "-I{}",
            build_dir.join("_deps/qdldl-build/include").display()
        ));
    let bindings = builder
        .generate()
        .expect("failed to generate embedded OSQP bindings");
    bindings
        .write_to_file(out_dir.join("bindings_embedded_raw.rs"))
        .expect("failed to write embedded OSQP bindings");
}

fn compile_generated_solver(generated_root: &Path, src_dir: &Path) {
    let workspace_c = require_generated_file(
        generated_root,
        generated_solver_contract::GENERATED_SOLVER_WORKSPACE_SOURCE,
    );
    let workspace_h = require_generated_file(
        generated_root,
        &format!(
            "{}/{}",
            generated_solver_contract::GENERATED_SOLVER_INCLUDE_DIR,
            generated_solver_contract::GENERATED_SOLVER_WORKSPACE_HEADER
        ),
    );
    let types_h = require_generated_file(
        generated_root,
        &format!(
            "{}/{}",
            generated_solver_contract::GENERATED_SOLVER_INCLUDE_DIR,
            generated_solver_contract::GENERATED_SOLVER_TYPES_HEADER
        ),
    );
    let qdldl_interface_h = require_generated_file(
        generated_root,
        &format!(
            "{}/{}",
            generated_solver_contract::GENERATED_SOLVER_INCLUDE_DIR,
            generated_solver_contract::GENERATED_SOLVER_QDLDL_INTERFACE_HEADER
        ),
    );
    let osqp_configure_h = require_generated_file(
        generated_root,
        &format!(
            "{}/{}",
            generated_solver_contract::GENERATED_SOLVER_CONFIGURE_DIR,
            generated_solver_contract::GENERATED_SOLVER_CONFIGURE_HEADER
        ),
    );

    println!("cargo:rerun-if-changed={}", workspace_c.display());
    println!("cargo:rerun-if-changed={}", workspace_h.display());
    println!("cargo:rerun-if-changed={}", types_h.display());
    println!("cargo:rerun-if-changed={}", qdldl_interface_h.display());
    println!("cargo:rerun-if-changed={}", osqp_configure_h.display());

    cc::Build::new()
        .define("EMBEDDED", "2")
        .file(&workspace_c)
        .include(generated_root.join(generated_solver_contract::GENERATED_SOLVER_INCLUDE_DIR))
        .include(generated_root.join(generated_solver_contract::GENERATED_SOLVER_CONFIGURE_DIR))
        .include(generated_root.join(generated_solver_contract::GENERATED_SOLVER_SOURCE_DIR))
        .include(src_dir)
        .include(src_dir.join("include"))
        .include(src_dir.join("include/public"))
        .include(src_dir.join("include/private"))
        .include(src_dir.join("algebra/_common"))
        .include(src_dir.join("algebra/_common/lin_sys/qdldl"))
        .compile("osqp_generated_solver");
}

fn validate_generated_solver_layout(generated_root: &Path) {
    require_generated_file(
        generated_root,
        generated_solver_contract::GENERATED_SOLVER_WORKSPACE_SOURCE,
    );
    require_generated_file(
        generated_root,
        &format!(
            "{}/{}",
            generated_solver_contract::GENERATED_SOLVER_INCLUDE_DIR,
            generated_solver_contract::GENERATED_SOLVER_WORKSPACE_HEADER
        ),
    );
    require_generated_file(
        generated_root,
        &format!(
            "{}/{}",
            generated_solver_contract::GENERATED_SOLVER_INCLUDE_DIR,
            generated_solver_contract::GENERATED_SOLVER_TYPES_HEADER
        ),
    );
    require_generated_file(
        generated_root,
        &format!(
            "{}/{}",
            generated_solver_contract::GENERATED_SOLVER_INCLUDE_DIR,
            generated_solver_contract::GENERATED_SOLVER_QDLDL_INTERFACE_HEADER
        ),
    );
    require_generated_file(
        generated_root,
        &format!(
            "{}/{}",
            generated_solver_contract::GENERATED_SOLVER_CONFIGURE_DIR,
            generated_solver_contract::GENERATED_SOLVER_CONFIGURE_HEADER
        ),
    );
}

fn require_generated_file(root: &Path, relative: &str) -> PathBuf {
    let path = root.join(relative);
    if !path.is_file() {
        panic!(
            "{}={:?} is missing required generated artifact `{}`; expected a directory produced by OSQP 0.6.3 code generation",
            generated_solver_contract::GENERATED_SOLVER_ENV,
            root,
            relative
        );
    }
    path
}
