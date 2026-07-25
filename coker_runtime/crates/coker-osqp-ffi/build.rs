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
    println!("cargo:rerun-if-env-changed=CARGO_CFG_TARGET_ARCH");
    println!("cargo:rerun-if-env-changed=CARGO_CFG_TARGET_ABI");
    println!("cargo:rerun-if-env-changed=CARGO_CFG_TARGET_ENDIAN");
    println!("cargo:rerun-if-env-changed=COKER_OSQP_CMAKE_TOOLCHAIN_FILE");
    println!("cargo:rerun-if-changed=cmake/thumbv7em-none-eabihf.cmake");
    println!("cargo:rerun-if-env-changed=CARGO_CFG_TARGET_POINTER_WIDTH");
    println!("cargo:rerun-if-changed=coker_osqp_abi.c");
    println!("cargo:rerun-if-changed=coker_osqp_abi.h");

    println!("cargo:rustc-check-cfg=cfg(osqp_dlong)");
    println!("cargo:rustc-check-cfg=cfg(osqp_embedded)");
    println!("cargo:rustc-check-cfg=cfg(osqp_generated_solver)");

    if !Path::new("osqp/README.md").exists() {
        let _ = Command::new("git")
            .args(["submodule", "update", "--init", "--recursive"])
            .status();
    }

    let embedded = env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("none");
    if embedded {
        require_supported_embedded_target();
    }

    let generated_solver_dir =
        env::var_os(generated_solver_contract::GENERATED_SOLVER_ENV).map(PathBuf::from);

    if generated_solver_dir.is_some() && !embedded {
        panic!(
            "{} requires target_os=none because the supplied OSQP 0.6.3 generated artifacts are built against the embedded runtime.",
            generated_solver_contract::GENERATED_SOLVER_ENV
        );
    }
    if let Some(generated_solver_dir) = generated_solver_dir.as_ref() {
        validate_generated_solver_layout(generated_solver_dir);
    }

    let dlong_enabled = if embedded {
        "OFF"
    } else {
        match &*env::var("CARGO_CFG_TARGET_POINTER_WIDTH").unwrap() {
            "64" => {
                println!("cargo:rustc-cfg=osqp_dlong");
                "ON"
            }
            "32" => "OFF",
            other => panic!(
                "{} bit targets are not supported. If you want this feature please file a bug.",
                other
            ),
        }
    };

    if embedded {
        println!("cargo:rustc-cfg=osqp_embedded");
    }
    if generated_solver_dir.is_some() {
        println!("cargo:rustc-cfg=osqp_generated_solver");
    }

    // The CMake build script for OSQP generates files inside the source directory.
    // The docs.rs builder does not like this, so we copy the OSQP source tree into `OUT_DIR`.
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
    if embedded {
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
    config.define("CTRLC", "OFF").define("UNITTESTS", "OFF");
    if embedded {
        configure_embedded_osqp(&mut config);
    } else {
        config
            .define("DFLOAT", "OFF")
            .define("DLONG", dlong_enabled)
            .define("ENABLE_MKL_PARDISO", "OFF")
            .define("PRINTING", "OFF")
            .define("PROFILING", "OFF");
    }

    config.build_target("osqpstatic").build();

    if embedded {
        compile_embedded_bridge(&src_dir, &build_dir);
        compile_embedded_abi(&src_dir, &build_dir);
    }
    if let Some(generated_solver_dir) = generated_solver_dir.as_ref() {
        compile_generated_solver(generated_solver_dir, &src_dir);
    }

    let native_lib_dir = if embedded {
        build_dir.join("out")
    } else if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
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
    println!("cargo:rustc-link-lib=static=osqp");
}

fn require_supported_embedded_target() {
    require_target_cfg("CARGO_CFG_TARGET_ARCH", "arm");
    require_target_cfg("CARGO_CFG_TARGET_ABI", "eabihf");
    require_target_cfg("CARGO_CFG_TARGET_ENDIAN", "little");
    require_target_cfg("CARGO_CFG_TARGET_POINTER_WIDTH", "32");
}

fn require_target_cfg(name: &str, expected: &str) {
    let actual = env::var(name).unwrap_or_default();
    if actual != expected {
        panic!(
            "target_os=none OSQP requires {}={} for its f32/i32/32-bit-pointer ABI; got {}={:?}",
            name, expected, name, actual
        );
    }
}

fn configure_embedded_osqp(config: &mut Config) {
    config
        .define("EMBEDDED", "2")
        .define("DFLOAT", "ON")
        .define("DLONG", "OFF")
        .define("ENABLE_MKL_PARDISO", "OFF")
        .define("PRINTING", "OFF")
        .define("PROFILING", "OFF");
}

fn compile_embedded_bridge(src_dir: &Path, build_dir: &Path) {
    cc::Build::new()
        .std("c99")
        .define("EMBEDDED", "2")
        .file("embedded_bridge.c")
        .include(src_dir)
        .include(src_dir.join("include"))
        .include(src_dir.join("include/public"))
        .include(src_dir.join("include/private"))
        .include(build_dir)
        .include(build_dir.join("include"))
        .include(build_dir.join("include/public"))
        .include(build_dir.join("include/private"))
        .include(build_dir.join("_deps/qdldl-build/include"))
        .include(src_dir.join("lin_sys/direct/qdldl/qdldl_sources/include"))
        .compile("osqp_embedded_bridge");
}

fn compile_embedded_abi(src_dir: &Path, build_dir: &Path) {
    cc::Build::new()
        .std("c99")
        .define("EMBEDDED", "2")
        .file("coker_osqp_abi.c")
        .include(src_dir)
        .include(src_dir.join("include"))
        .include(src_dir.join("include/public"))
        .include(src_dir.join("include/private"))
        .include(build_dir)
        .include(build_dir.join("include"))
        .include(build_dir.join("include/public"))
        .include(build_dir.join("include/private"))
        .include(build_dir.join("_deps/qdldl-build/include"))
        .include(src_dir.join("lin_sys/direct/qdldl/qdldl_sources/include"))
        .compile("coker_osqp_abi");
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
        .std("c99")
        .define("EMBEDDED", "2")
        .file(&workspace_c)
        .include(generated_root.join(generated_solver_contract::GENERATED_SOLVER_INCLUDE_DIR))
        .include(generated_root.join(generated_solver_contract::GENERATED_SOLVER_CONFIGURE_DIR))
        .include(generated_root.join(generated_solver_contract::GENERATED_SOLVER_SOURCE_DIR))
        .include(src_dir)
        .include(src_dir.join("include"))
        .include(src_dir.join("include/public"))
        .include(src_dir.join("include/private"))
        .include(src_dir.join("lin_sys/direct/qdldl/qdldl_sources/include"))
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
