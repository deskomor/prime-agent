//! Where a session's `local` harness store lives: one store per project,
//! shared by every session whose working directory belongs to it.
//!
//! Fork-only behaviour: upstream keeps the local store in the session's
//! artifact dir. The wire name stays `local`.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use super::HARNESS_STATE_DIR_NAME;

/// Directory under the agent dir holding one folder per project.
pub const PROJECTS_DIR_NAME: &str = "projects";

/// File inside a session's own harness dir that pins the session to that
/// store instead of its project's: a `/refine --probe` child runs on its
/// seed and must neither read nor write the real project store.
pub const SESSION_STORE_MARKER_FILE_NAME: &str = "session_store";

/// The directory that identifies the project `cwd` belongs to: the checkout
/// that owns the git common dir (so every worktree of one repository is the
/// same project), else `cwd` itself. Read from the filesystem, no `git`.
#[must_use]
pub fn project_root(cwd: &Path) -> PathBuf {
    let cwd = cwd.canonicalize().unwrap_or_else(|_| cwd.to_path_buf());
    for dir in cwd.ancestors() {
        let dot_git = dir.join(".git");
        if dot_git.is_dir() {
            return dir.to_path_buf();
        }
        // A linked worktree or submodule: `.git` is a file naming the git dir.
        let Ok(pointer) = std::fs::read_to_string(&dot_git) else {
            continue;
        };
        let Some(git_dir) = pointer.trim().strip_prefix("gitdir:") else {
            continue;
        };
        let git_dir = dir.join(git_dir.trim());
        let common = std::fs::read_to_string(git_dir.join("commondir")).map_or_else(
            |_| git_dir.clone(),
            |relative| git_dir.join(relative.trim()),
        );
        let common = common.canonicalize().unwrap_or(common);
        return match (common.file_name(), common.parent()) {
            (Some(name), Some(parent)) if name == ".git" => parent.to_path_buf(),
            _ => common,
        };
    }
    cwd
}

/// The project's folder name under `projects/`: the root's own name (made
/// filesystem-safe) plus a short hash of its full path.
#[must_use]
pub fn project_key(cwd: &Path) -> String {
    let root = project_root(cwd);
    let name: String = root
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_default()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let digest = Sha256::digest(root.to_string_lossy().as_bytes());
    let hash = digest.iter().take(6).fold(String::new(), |mut hash, byte| {
        use std::fmt::Write;
        let _ = write!(hash, "{byte:02x}");
        hash
    });
    if name.is_empty() {
        hash
    } else {
        format!("{name}-{hash}")
    }
}

/// `<agent dir>/projects/<key>/harness`: state, refinement history and
/// verdict log of the project `cwd` belongs to.
#[must_use]
pub fn get_project_harness_state_dir(agent_dir: &Path, cwd: &Path) -> PathBuf {
    agent_dir
        .join(PROJECTS_DIR_NAME)
        .join(project_key(cwd))
        .join(HARNESS_STATE_DIR_NAME)
}

/// The store a session's `local` scope reads and writes: its project's,
/// unless the session is pinned to its own (see
/// [`SESSION_STORE_MARKER_FILE_NAME`]). `None` for a session without an
/// artifact dir, which has no local scope.
#[must_use]
pub fn resolve_local_harness_state_dir(
    agent_dir: &Path,
    cwd: &Path,
    session_artifact_dir: Option<&Path>,
) -> Option<PathBuf> {
    let own = super::get_local_harness_state_dir(session_artifact_dir)?;
    Some(if own.join(SESSION_STORE_MARKER_FILE_NAME).exists() {
        own
    } else {
        get_project_harness_state_dir(agent_dir, cwd)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worktrees_of_one_repository_share_a_store_and_a_pinned_session_keeps_its_own() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let agent = root.join("agent");
        // A main checkout, and a linked worktree laid out the way git writes one.
        let repo = root.join("my repo");
        let linked_git_dir = repo.join(".git/worktrees/wt");
        std::fs::create_dir_all(&linked_git_dir).unwrap();
        std::fs::write(linked_git_dir.join("commondir"), "../..\n").unwrap();
        let worktree = root.join("elsewhere/wt");
        std::fs::create_dir_all(worktree.join("src")).unwrap();
        std::fs::write(
            worktree.join(".git"),
            format!("gitdir: {}\n", linked_git_dir.display()),
        )
        .unwrap();
        let other = root.join("other");
        std::fs::create_dir_all(&other).unwrap();

        let from_main = get_project_harness_state_dir(&agent, &repo);
        assert_eq!(
            from_main.strip_prefix(&agent).unwrap().iter().count(),
            3,
            "projects/<key>/harness"
        );
        assert!(project_key(&repo).starts_with("my_repo-"));
        assert_eq!(
            get_project_harness_state_dir(&agent, &worktree.join("src")),
            from_main
        );
        assert_ne!(get_project_harness_state_dir(&agent, &other), from_main);

        let artifacts = root.join("session-artifacts/s1");
        assert_eq!(
            resolve_local_harness_state_dir(&agent, &repo, Some(&artifacts)),
            Some(from_main)
        );
        let own = artifacts.join(HARNESS_STATE_DIR_NAME);
        std::fs::create_dir_all(&own).unwrap();
        std::fs::write(own.join(SESSION_STORE_MARKER_FILE_NAME), "").unwrap();
        assert_eq!(
            resolve_local_harness_state_dir(&agent, &repo, Some(&artifacts)),
            Some(own)
        );
        assert_eq!(resolve_local_harness_state_dir(&agent, &repo, None), None);
    }
}
