//! Immutable source identity compiled into the daemon and its CLI consumer.
use std::path::{Path, PathBuf};
use std::process::Command;

fn git(directory: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .env("GIT_OPTIONAL_LOCKS", "0")
        .args(args)
        .current_dir(directory)
        .output()
        .ok()?;
    output.status.success().then_some(())?;
    String::from_utf8(output.stdout).ok()
}

fn watch_git_path(directory: &Path, name: &str) {
    if let Some(path) = git(directory, &["rev-parse", "--git-path", name]) {
        let path = PathBuf::from(path.trim());
        let path = if path.is_absolute() {
            path
        } else {
            directory.join(path)
        };
        println!("cargo:rerun-if-changed={}", path.display());
    }
}

fn main() {
    println!("cargo:rerun-if-env-changed=PRIME_AGENT_CLEAN_BUILD_COMMIT");
    let clean_commit = std::env::var("PRIME_AGENT_CLEAN_BUILD_COMMIT").ok();
    let manifest =
        PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").expect("Cargo manifest dir"));
    let workspace = manifest.join("../..").canonicalize().ok();
    let root = git(&manifest, &["rev-parse", "--show-toplevel"])
        .and_then(|root| PathBuf::from(root.trim()).canonicalize().ok())
        .filter(|root| Some(root) == workspace.as_ref());
    let mut commit = None;
    let mut dirty = None;
    if let Some(root) = root {
        for name in ["HEAD", "index", "packed-refs"] {
            watch_git_path(&root, name);
        }
        if let Some(branch) = git(&root, &["symbolic-ref", "--quiet", "HEAD"]) {
            watch_git_path(&root, branch.trim());
        }
        // Tracked input edits can change dirty state without touching the Git index.
        if let Some(files) = git(
            &root,
            &[
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
            ],
        ) {
            for file in files.split('\0').filter(|file| !file.is_empty()) {
                println!("cargo:rerun-if-changed={}", root.join(file).display());
            }
        }
        commit = git(&root, &["rev-parse", "--verify", "HEAD"])
            .map(|head| head.trim().to_string())
            .filter(|head| head.len() == 40 && head.bytes().all(|byte| byte.is_ascii_hexdigit()));
        dirty = git(
            &root,
            &["status", "--porcelain=v1", "--untracked-files=normal"],
        )
        .and_then(|status| {
            if !status.is_empty() {
                Some(true)
            } else if commit.is_some() && clean_commit.as_deref() == commit.as_deref() {
                Some(false)
            } else {
                None
            }
        });
    }
    println!(
        "cargo:rustc-env=PA_BUILD_COMMIT={}",
        commit.as_deref().unwrap_or("unknown")
    );
    println!(
        "cargo:rustc-env=PA_BUILD_DIRTY={}",
        dirty.map_or("unknown", |dirty| if dirty { "true" } else { "false" })
    );
}
