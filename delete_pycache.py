#!/usr/bin/env python3
"""
APGI Validation Cleanup Script

This script removes temporary files and output directories generated during validation processes.

Features:
- Removes Python cache files (__pycache__, *.pyc, etc.)
- Cleans APGI-specific output directories (apgi_output, apgi_complete_output, etc.)
- Removes visualization files (*.png, *.html, *.svg)
- Cleans debugging and temporary files (debug_*, test_*, temp_*, etc.)
- Preserves core functionality and important project files
- Supports dry-run mode for safe testing
- APGI-specific options for selective cleanup

Usage Examples:
  python delete_pycache.py                    # Standard cleanup
  python delete_pycache.py --dry-run          # Preview what will be removed
  python delete_pycache.py --apgi-only        # Only APGI-specific files
  python delete_pycache.py --keep-visualizations  # Keep *.png, *.html, *.svg
  python delete_pycache.py --keep-reports     # Keep *.json, *.html reports
"""

import argparse
import errno
import fnmatch
import os
import shutil
import sys
import time
from typing import Any, Iterable, List, Optional

DEFAULT_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".hypothesis",
    "htmlcov",
    ".tox",
    ".ipynb_checkpoints",
    ".cache",
    "build",
    "dist",
    "backups",
    ".coverage",
    "site-packages",
    "cache",
}

DEFAULT_DIR_PATTERNS = ["*.egg-info", "pip-wheel-metadata"]

DEFAULT_FILE_PATTERNS = [
    "*.pyc",
    "*.pyo",
    "*.pyd",
    ".coverage",
    "coverage.xml",
    ".coverage.*",
    "*.log",
    ".DS_Store",
    "Thumbs.db",
    "*.pkl",
    "*.pickle",
    "*.npz",
    "*.npy",
    "*.h5",
    "*.hdf5",
    "*.mat",
    "*.parquet",
    "*.feather",
    "*.arrow",
    "*.avro",
    "*.orc",
    "*.msgpack",
    "*.bson",
    "*.db",
    "*.sqlite",
    "*.sqlite3",
    "*.mdb",
    "*.ldb",
    "*.pak",
    "*.pack",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.safetensors",
    "*.onnx",
    "*.tflite",
    "*.pb",
    "*.plan",
    "*.engine",
    "*.tfevents.*",
    "events.out.tfevents.*",
    "*.v2",
    "*.proto.bin",
    "*_pb2.py",
    "*_pb2.pyi",
    "*.proto.lua",
    "*.nbconvert.ipynb",
    "*.ipynb_checkpoints",
    "*.aux",
    "*.bbl",
    "*.blg",
    "*.fdb_latexmk",
    "*.fls",
    "*.synctex.gz",
    "*.toc",
    "*.snm",
    "*.nav",
    "*.vrb",
    "*.figlist",
    "*.makefile",
    "*.xdv",
    "*.run.xml",
    "debug_*.py",
    "temp_*.py",
    "*_temp.py",
    "*_debug.py",
    "*_backup.py",
    "*_copy.py",
    "*_old.py",
    "*_orig.py",
    "*_tmp.py",
    "*.generated.*",
    "*.auto.*",
    "*.cache.json",
    "*.cache.yaml",
    "*.cache.yml",
    "*.min.json",
    "*.min.yaml",
    "*.min.yml",
    "*.manifest",
    "*.spec",
    "*.egg",
    "*.whl",
    "*.so",
    "*.dylib",
    "*.dll",
    "*.a",
    "*.lib",
    "*.o",
    "*.obj",
    "*.intermediate",
    "*.partial",
    "*.incomplete",
    "*.processing",
    "*.tmp.json",
    "*.tmp.csv",
    "*.tmp.parquet",
    "*.download",
    "*.crdownload",
    "*.part",
]

DEFAULT_EXTRA_DIR_NAMES = {
    ".nox",
    ".ruff_cache",
    ".benchmarks",
    "screenshots",
    "temp",
    "tmp",
    "figures",
    "plots",
    "debug_output",
}

DEFAULT_SKIP_TRAVERSE_DIRS = {".git", ".svn", ".hg"}

DEFAULT_PROTECTED_DIR_NAMES = {
    "tests",
    "test",
    "src",
    "source",
    "docs",
    "doc",
    "examples",
    "example",
    "scripts",
    "bin",
    "lib",
    "utils",
    "Validation",
    "Falsification",
    "Theory",
    "config",
    "data",
    "data_repository",
    ".keys",
    ".kiro",
}

DEFAULT_EXTRA_FILE_PATTERNS = [
    "*.tmp",
    "*.temp",
    "*~",
    "*.swp",
    "*.swo",
    "*.swn",
    "*.un~",
    "*.bak",
    "*.orig",
    ".#*",
    "#*#",
    "*.debug",
    "*.test",
    "*.old",
    "*.copy",
    "*.backup",
    "*.save",
    "*.saved",
    "*.new",
    "*.rej",
    "*.patch",
    "*.diff",
    "*.merge",
    "*.merge_left",
    "*.merge_right",
    "*.working",
    "*.base",
    "*.local",
    "*.remote",
    "*.mine",
    "*.theirs",
    "*.r*",
    "4913",
]


def matches_any(name: str, patterns: Iterable[str]) -> bool:
    for pat in patterns:
        if fnmatch.fnmatch(name, pat):
            return True
    return False


def _should_remove_directory(
    dirname: str,
    default_dir_names: set,
    default_dir_patterns: list,
    include_dir_patterns: Iterable[str],
    remove_node_modules: bool,
    remove_venvs: bool,
    venv_names: Iterable[str],
    protected_dir_names: set | None = None,
) -> bool:
    """Determine if a directory should be removed."""
    if protected_dir_names and dirname in protected_dir_names:
        return False

    return (
        dirname in default_dir_names
        or matches_any(dirname, default_dir_patterns)
        or matches_any(dirname, include_dir_patterns)
        or (remove_node_modules and dirname == "node_modules")
        or (remove_venvs and dirname in set(venv_names))
    )


def _remove_directory(
    dirpath: str,
    dirname: str,
    dry_run: bool,
    verbose: bool,
    stats: dict,
    max_retries: int = 3,
    initial_delay: float = 0.1,
) -> None:
    """Remove a directory and update stats with retry logic for concurrent access."""
    full_d = os.path.join(dirpath, dirname)
    if dry_run:
        if verbose:
            print(f"Would remove directory: {full_d}")
    else:
        for attempt in range(max_retries):
            try:
                shutil.rmtree(full_d, ignore_errors=False)
                stats["dirs_removed"] += 1
                if verbose:
                    print(f"Removed directory: {full_d}")
                break
            except OSError as e:
                if (
                    e.errno in (errno.EBUSY, errno.EACCES, errno.ENOTEMPTY)
                    and attempt < max_retries - 1
                ):
                    delay = initial_delay * (2**attempt)
                    if verbose:
                        print(
                            f"Concurrent access detected for {full_d}, retrying in {delay:.2f}s..."
                        )
                    time.sleep(delay)
                else:
                    stats["errors"] += 1
                    print(f"Error removing directory {full_d}: {e}")
                    break
            except Exception as e:
                stats["errors"] += 1
                print(f"Error removing directory {full_d}: {e}")
                break


def _remove_file(
    filepath: str,
    dry_run: bool,
    verbose: bool,
    stats: dict,
    max_retries: int = 3,
    initial_delay: float = 0.1,
) -> None:
    """Remove a file and update stats with retry logic for concurrent access."""
    if dry_run:
        if verbose:
            print(f"Would remove file: {filepath}")
    else:
        for attempt in range(max_retries):
            try:
                os.remove(filepath)
                stats["files_removed"] += 1
                if verbose:
                    print(f"Removed file: {filepath}")
                break
            except OSError as e:
                if (
                    e.errno in (errno.EBUSY, errno.EACCES, errno.ENOENT)
                    and attempt < max_retries - 1
                ):
                    delay = initial_delay * (2**attempt)
                    if verbose:
                        print(
                            f"Concurrent access detected for {filepath}, retrying in {delay:.2f}s..."
                        )
                    time.sleep(delay)
                elif e.errno == errno.ENOENT:
                    stats["files_removed"] += 1
                    if verbose:
                        print(f"File already removed: {filepath}")
                    break
                else:
                    stats["errors"] += 1
                    print(f"Error removing file {filepath}: {e}")
                    break
            except Exception as e:
                stats["errors"] += 1
                print(f"Error removing file {filepath}: {e}")
                break


def _process_directories(
    dirpath: str,
    dirnames: list,
    default_dir_names: set,
    default_dir_patterns: list,
    include_dir_patterns: Iterable[str],
    exclude_dir_patterns: Iterable[str],
    remove_node_modules: bool,
    remove_venvs: bool,
    venv_names: Iterable[str],
    dry_run: bool,
    verbose: bool,
    stats: dict,
    protected_dir_names: set | None = None,
) -> None:
    """Process directories in current path."""
    for d in list(dirnames):
        if d in DEFAULT_SKIP_TRAVERSE_DIRS or matches_any(d, exclude_dir_patterns):
            dirnames.remove(d)
            continue

        should_remove = _should_remove_directory(
            d,
            default_dir_names,
            default_dir_patterns,
            include_dir_patterns,
            remove_node_modules,
            remove_venvs,
            venv_names,
            protected_dir_names,
        )

        if should_remove:
            _remove_directory(dirpath, d, dry_run, verbose, stats)
            if d in dirnames:
                dirnames.remove(d)
            continue

        if protected_dir_names and d in protected_dir_names:
            continue


def _process_files(
    dirpath: str,
    filenames: list,
    default_file_patterns: list,
    include_file_patterns: Iterable[str],
    exclude_file_patterns: Iterable[str],
    dry_run: bool,
    verbose: bool,
    stats: dict,
    protected_dir_names: set | None = None,
) -> None:
    """Process files in current path."""
    cache_patterns = {"*.pyc", "*.pyo", "*.pyd", ".coverage", "*.log"}

    for f in list(filenames):
        if matches_any(f, exclude_file_patterns):
            continue

        is_cache_file = matches_any(f, cache_patterns)
        is_protected_path = False
        if protected_dir_names:
            path_parts = dirpath.split(os.sep)
            is_protected_path = any(part in protected_dir_names for part in path_parts)

        if is_protected_path and not is_cache_file:
            continue

        if matches_any(f, default_file_patterns) or matches_any(
            f, include_file_patterns
        ):
            full_f = os.path.join(dirpath, f)
            _remove_file(full_f, dry_run, verbose, stats)


def _should_skip_directory(
    dirpath: str, root_dir: str, max_depth: Optional[int]
) -> bool:
    """Check if directory should be skipped based on depth."""
    if max_depth is None:
        return False
    rel = os.path.relpath(dirpath, root_dir)
    depth = 0 if rel == "." else rel.count(os.sep) + 1
    if depth > max_depth:
        return True
    return False


def preview_deletions(
    root_dir: str,
    include_dir_patterns: Iterable[str] = (),
    include_file_patterns: Iterable[str] = (),
    exclude_dir_patterns: Iterable[str] = (),
    exclude_file_patterns: Iterable[str] = (),
    remove_node_modules: bool = False,
    remove_venvs: bool = False,
    venv_names: Iterable[str] = (".venv", "venv", ".env", "env"),
    follow_links: bool = False,
    max_depth: Optional[int] = None,
) -> dict:
    """Preview what would be deleted without actually deleting anything.

    Returns detailed information about files and directories that would be removed,
    including file sizes and directory counts for better decision making.
    """
    stats: dict[str, Any] = {
        "dirs_to_remove": [],
        "files_to_remove": [],
        "total_files": 0,
        "total_size_bytes": 0,
        "errors": 0,
    }

    default_dir_names = set(DEFAULT_DIR_NAMES) | set(DEFAULT_EXTRA_DIR_NAMES)
    default_dir_patterns = list(DEFAULT_DIR_PATTERNS)
    default_file_patterns = list(DEFAULT_FILE_PATTERNS) + list(
        DEFAULT_EXTRA_FILE_PATTERNS
    )

    for dirpath, dirnames, filenames in os.walk(
        root_dir, topdown=True, followlinks=follow_links
    ):
        if _should_skip_directory(dirpath, root_dir, max_depth):
            dirnames[:] = []
            continue

        # Process directories for preview
        for d in list(dirnames):
            if d in DEFAULT_SKIP_TRAVERSE_DIRS or matches_any(d, exclude_dir_patterns):
                continue

            if _should_remove_directory(
                d,
                default_dir_names,
                default_dir_patterns,
                include_dir_patterns,
                remove_node_modules,
                remove_venvs,
                venv_names,
            ):
                full_d = os.path.join(dirpath, d)
                try:
                    # Count files and calculate size in directory
                    dir_size = 0
                    file_count = 0
                    for root, dirs, files in os.walk(full_d):
                        for file in files:
                            file_path = os.path.join(root, file)
                            try:
                                file_size = os.path.getsize(file_path)
                                dir_size += file_size
                                file_count += 1
                            except (OSError, IOError):
                                stats["errors"] += 1  # type: ignore[index,operator]

                    stats["dirs_to_remove"].append(  # type: ignore[attr-defined]
                        {
                            "path": full_d,
                            "file_count": file_count,
                            "size_bytes": dir_size,
                            "size_mb": round(dir_size / (1024 * 1024), 2),
                        }
                    )
                    stats["total_files"] += file_count  # type: ignore[index,operator]
                    stats["total_size_bytes"] += dir_size  # type: ignore[index,operator]
                except (OSError, IOError):
                    stats["errors"] += 1  # type: ignore[index,operator]

        # Process files for preview
        for f in list(filenames):
            if matches_any(f, exclude_file_patterns):
                continue

            if matches_any(f, default_file_patterns) or matches_any(
                f, include_file_patterns
            ):
                full_f = os.path.join(dirpath, f)
                try:
                    file_size = os.path.getsize(full_f)
                    stats["files_to_remove"].append(  # type: ignore[attr-defined]
                        {
                            "path": full_f,
                            "size_bytes": file_size,
                            "size_kb": round(file_size / 1024, 2),
                        }
                    )
                    stats["total_files"] += 1  # type: ignore[index,operator]
                    stats["total_size_bytes"] += file_size  # type: ignore[index,operator]
                except (OSError, IOError):
                    stats["errors"] += 1  # type: ignore[index,operator]

    return stats


def format_preview(stats: dict, verbose: bool = True) -> None:
    """Format and display deletion preview in a user-friendly way."""
    if verbose:
        print("\n" + "=" * 60)
        print("DELETION PREVIEW")
        print("=" * 60)

        # Summary
        total_size_mb = round(stats["total_size_bytes"] / (1024 * 1024), 2)
        print("\nSUMMARY:")
        print(f"  Files to delete: {len(stats['files_to_remove'])}")
        print(f"  Directories to delete: {len(stats['dirs_to_remove'])}")
        print(f"  Total files affected: {stats['total_files']}")
        print(f"  Total space to free: {total_size_mb} MB")

        if stats["errors"] > 0:
            print(f"  Errors encountered: {stats['errors']}")

        # Directories
        if stats["dirs_to_remove"]:
            print(f"\nDIRECTORIES TO DELETE ({len(stats['dirs_to_remove'])}):")
            for i, dir_info in enumerate(stats["dirs_to_remove"][:10]):  # Show first 10
                print(f"  {i + 1}. {dir_info['path']}")
                print(
                    f"     Files: {dir_info['file_count']}, Size: {dir_info['size_mb']} MB"
                )

            if len(stats["dirs_to_remove"]) > 10:
                print(f"  ... and {len(stats['dirs_to_remove']) - 10} more directories")

        # Files (show largest ones)
        if stats["files_to_remove"]:
            # Sort files by size (largest first)
            sorted_files = sorted(
                stats["files_to_remove"], key=lambda x: x["size_bytes"], reverse=True
            )

            print(
                f"\nLARGEST FILES TO DELETE (showing top 10 of {len(stats['files_to_remove'])}):"
            )
            for i, file_info in enumerate(sorted_files[:10]):
                print(f"  {i + 1}. {file_info['path']}")
                print(f"     Size: {file_info['size_kb']} KB")

        print("\n" + "=" * 60)
        print("Use --dry-run to see this preview without deleting")
        print("Use --yes to proceed with deletion")
        print("=" * 60)


def delete_temporary_items(
    root_dir: str,
    dry_run: bool = False,
    verbose: bool = True,
    include_dir_patterns: Iterable[str] = (),
    include_file_patterns: Iterable[str] = (),
    exclude_dir_patterns: Iterable[str] = (),
    exclude_file_patterns: Iterable[str] = (),
    remove_node_modules: bool = False,
    remove_venvs: bool = False,
    venv_names: Iterable[str] = (".venv", "venv", ".env", "env"),
    follow_links: bool = False,
    max_depth: Optional[int] = None,
) -> dict:
    """Delete common temporary directories and files under root_dir.

    - Removes directories in DEFAULT_DIR_NAMES and those matching DEFAULT_DIR_PATTERNS.
    - Removes files matching DEFAULT_FILE_PATTERNS.
    - Removes directories matching patterns (like '*.egg-info').
    - Specifically tailored for APGI validation app cleanup.

    This function avoids descending into removed directories by modifying dirnames in-place.
    Returns statistics about what was removed.
    """
    stats: dict[str, Any] = {"dirs_removed": 0, "files_removed": 0, "errors": 0}
    default_dir_names = set(DEFAULT_DIR_NAMES) | set(DEFAULT_EXTRA_DIR_NAMES)
    default_dir_patterns = list(DEFAULT_DIR_PATTERNS)
    default_file_patterns = list(DEFAULT_FILE_PATTERNS) + list(
        DEFAULT_EXTRA_FILE_PATTERNS
    )
    protected_dir_names = DEFAULT_PROTECTED_DIR_NAMES

    for dirpath, dirnames, filenames in os.walk(
        root_dir, topdown=True, followlinks=follow_links
    ):
        if _should_skip_directory(dirpath, root_dir, max_depth):
            dirnames[:] = []
            continue

        _process_directories(
            dirpath,
            dirnames,
            default_dir_names,
            default_dir_patterns,
            include_dir_patterns,
            exclude_dir_patterns,
            remove_node_modules,
            remove_venvs,
            venv_names,
            dry_run,
            verbose,
            stats,
            protected_dir_names,
        )

        _process_files(
            dirpath,
            filenames,
            default_file_patterns,
            include_file_patterns,
            exclude_file_patterns,
            dry_run,
            verbose,
            stats,
            protected_dir_names,
        )

    return stats


def prune_empty_dirs(
    root_dir: str, dry_run: bool = False, verbose: bool = True
) -> None:
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        # don't prune the root itself
        if dirpath == root_dir:
            continue
        try:
            if not os.listdir(dirpath):
                if dry_run:
                    if verbose:
                        print(f"Would remove empty directory: {dirpath}")
                else:
                    os.rmdir(dirpath)
                    if verbose:
                        print(f"Removed empty directory: {dirpath}")
        except Exception as e:
            print(f"Error pruning directory {dirpath}: {e}")


def _remove_logs_directory(log_dir: str, dry_run: bool, verbose: bool) -> None:
    """Remove the entire logs directory."""
    if dry_run:
        if verbose:
            print(f"Would remove logs directory: {log_dir}")
    else:
        try:
            shutil.rmtree(log_dir, ignore_errors=False)
            if verbose:
                print(f"Removed logs directory: {log_dir}")
        except Exception as e:
            print(f"Error removing logs directory {log_dir}: {e}")


def _truncate_log_file(file_path: str, dry_run: bool, verbose: bool) -> None:
    """Truncate a log file to 0 bytes."""
    try:
        if dry_run:
            if verbose:
                print(f"Would clear file: {file_path}")
        else:
            with open(file_path, "w", encoding="utf-8"):
                pass
            if verbose:
                print(f"Cleared: {file_path}")
    except Exception as e:
        print(f"Error clearing {file_path}: {str(e)}")


def clear_log_files(
    root_dir: str,
    delete_logs_dir: bool = False,
    dry_run: bool = False,
    verbose: bool = True,
) -> None:
    """Either truncate files under a `logs` dir, or delete the logs directory entirely.

    - If delete_logs_dir is True, the whole logs directory is removed.
    - If False, each file is truncated to 0 bytes.
    """
    log_dir = os.path.join(root_dir, "logs")

    if not os.path.exists(log_dir):
        if verbose:
            print(f"Log directory not found at: {log_dir}")
        return

    if delete_logs_dir:
        _remove_logs_directory(log_dir, dry_run, verbose)
        return

    for root, dirs, files in os.walk(log_dir):
        for file in files:
            file_path = os.path.join(root, file)
            _truncate_log_file(file_path, dry_run, verbose)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Remove temporary files and folders from APGI validation project tree"
    )
    p.add_argument(
        "root",
        nargs="?",
        default=None,
        help="Root directory to clean (defaults to script dir)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without deleting",
    )
    p.add_argument(
        "--preview",
        action="store_true",
        help="Show detailed preview of what would be deleted",
    )
    p.add_argument("--yes", action="store_true", help="Don't prompt for confirmation")
    p.add_argument(
        "--delete-logs",
        action="store_true",
        help="Remove the entire logs directory instead of truncating files",
    )
    p.add_argument("--quiet", action="store_true", help="Reduce output")

    # Advanced controls
    p.add_argument(
        "--include-dir",
        action="append",
        default=[],
        help="Additional directory patterns to remove (glob). Can be passed multiple times.",
    )
    p.add_argument(
        "--include-file",
        action="append",
        default=[],
        help="Additional file patterns to remove (glob). Can be passed multiple times.",
    )
    p.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        help="Directory patterns to exclude from deletion (glob). Can be passed multiple times.",
    )
    p.add_argument(
        "--exclude-file",
        action="append",
        default=[],
        help="File patterns to exclude from deletion (glob). Can be passed multiple times.",
    )
    p.add_argument(
        "--remove-node-modules",
        action="store_true",
        help="Also remove node_modules directories",
    )
    p.add_argument(
        "--remove-venvs",
        action="store_true",
        help="Also remove common virtualenv directories (.venv, venv, .env, env)",
    )
    p.add_argument(
        "--venv-names",
        nargs="*",
        default=None,
        help="Override names considered virtualenvs (space-separated)",
    )
    p.add_argument(
        "--follow-links",
        action="store_true",
        help="Follow symbolic links during traversal (use with caution)",
    )
    p.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Limit traversal depth relative to root (1 = only root level)",
    )
    p.add_argument(
        "--apgi-only",
        action="store_true",
        help="Only remove APGI-specific output files (apgi_*, cache, logs, etc.)",
    )
    p.add_argument(
        "--keep-visualizations",
        action="store_true",
        help="Keep visualization files (*.png, *.html, *.svg)",
    )
    p.add_argument(
        "--keep-reports",
        action="store_true",
        help="Keep report files (*.json, *.html)",
    )
    p.add_argument(
        "--prune-empty-dirs",
        action="store_true",
        help="Remove now-empty directories after cleanup",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_directory = os.path.abspath(args.root) if args.root else current_dir

    # Validate root directory exists
    if not os.path.exists(root_directory):
        print(f"Error: Root directory does not exist: {root_directory}")
        return 1

    if not os.path.isdir(root_directory):
        print(f"Error: Root path is not a directory: {root_directory}")
        return 1

    dry_run = args.dry_run or args.preview
    verbose = not args.quiet

    # Handle preview option
    if args.preview:
        preview_stats = preview_deletions(
            root_directory,
            include_dir_patterns=args.include_dir,
            include_file_patterns=args.include_file,
            exclude_dir_patterns=args.exclude_dir,
            exclude_file_patterns=args.exclude_file,
            remove_node_modules=args.remove_node_modules,
            remove_venvs=args.remove_venvs,
            venv_names=(
                args.venv_names
                if args.venv_names is not None
                else (".venv", "venv", ".env", "env")
            ),
            follow_links=args.follow_links,
            max_depth=args.max_depth,
        )
        format_preview(preview_stats, verbose=True)
        return 0

    # Handle APGI-specific options
    include_dir_patterns = list(args.include_dir)
    include_file_patterns = list(args.include_file)
    exclude_dir_patterns = list(args.exclude_dir)
    exclude_file_patterns = list(args.exclude_file)

    if args.apgi_only:
        # Only remove APGI-specific files and directories (complement to DEFAULT_EXTRA_*)
        include_dir_patterns.extend(
            [
                "apgi_*",
                "apgi_output*",
                "apgi_complete_output",
                "data_output",
                "analysis_output",
                "experiment_output",
                "validation_output",
                "testing_output",
                "debug_output",
            ]
        )
        include_file_patterns.extend(
            [
                "apgi_*",
                "debug_*",
                "temp_*",
                "*_temp.*",
                "*_debug.*",
                "*_backup.*",
                "*_copy.*",
                "*_old.*",
                "*_orig.*",
                # NOTE: "*_test.*" removed - would delete test files
                "*_tmp.*",
            ]
        )
        # Exclude core imperative files (source code) - keep all intact
        exclude_file_patterns.extend(
            [
                # Python source
                "*.py",
                "*.pyi",
                "*.pyx",
                "*.pxd",
                "*.pxi",
                "*.pyw",
                # JavaScript/TypeScript
                "*.js",
                "*.jsx",
                "*.ts",
                "*.tsx",
                "*.mjs",
                "*.cjs",
                "*.mts",
                "*.cts",
                # Web frontend
                "*.html",
                "*.htm",
                "*.css",
                "*.scss",
                "*.sass",
                "*.less",
                "*.vue",
                "*.svelte",
                # C/C++
                "*.c",
                "*.cpp",
                "*.cc",
                "*.cxx",
                "*.h",
                "*.hpp",
                "*.hh",
                "*.hxx",
                "*.inl",
                "*.inc",
                # Java/Kotlin/JVM
                "*.java",
                "*.kt",
                "*.kts",
                "*.scala",
                "*.groovy",
                "*.clj",
                "*.cljs",
                # .NET/C#
                "*.cs",
                "*.csx",
                "*.vb",
                "*.fs",
                "*.fsx",
                # Go/Rust
                "*.go",
                "*.rs",
                "*.rslib",
                # Ruby/Perl/PHP
                "*.rb",
                "*.rbw",
                "*.rake",
                "*.gemspec",
                "*.pl",
                "*.pm",
                "*.t",
                "*.php",
                "*.phtml",
                "*.php3",
                "*.php4",
                "*.php5",
                "*.phps",
                # Shell/Scripts
                "*.sh",
                "*.bash",
                "*.zsh",
                "*.fish",
                "*.csh",
                "*.tcsh",
                "*.ksh",
                "*.ps1",
                "*.psm1",
                "*.psd1",
                "*.bat",
                "*.cmd",
                # Functional
                "*.hs",
                "*.lhs",
                "*.elm",
                "*.erl",
                "*.hrl",
                "*.ex",
                "*.exs",
                "*.ml",
                "*.mli",
                "*.re",
                "*.rei",
                # Other languages
                "*.r",
                "*.R",
                "*.lua",
                "*.nim",
                "*.cr",
                "*.d",
                "*.jl",
                "*.swift",
                "*.m",
                "*.mm",
                "*.rkt",
                "*.lisp",
                "*.lsp",
                "*.scm",
                "*.ss",
                "*.f",
                "*.for",
                "*.f90",
                "*.f95",
                "*.f03",
                "*.f08",
                "*.cob",
                "*.cbl",
                "*.asm",
                "*.s",
                "*.S",
                # Config/Project files
                "requirements*.txt",
                "pyproject.toml",
                "setup.py",
                "setup.cfg",
                "Pipfile",
                "poetry.lock",
                "package.json",
                "package-lock.json",
                "yarn.lock",
                "pnpm-lock.yaml",
                "bun.lockb",
                "Cargo.toml",
                "Cargo.lock",
                "go.mod",
                "go.sum",
                "Gemfile",
                "Gemfile.lock",
                "composer.json",
                "composer.lock",
                "pom.xml",
                "build.gradle",
                "settings.gradle",
                "CMakeLists.txt",
                "Makefile",
                "makefile",
                "GNUmakefile",
                "*.mk",
                "*.pro",
                "*.pri",
                "*.cmake",
                # Documentation
                "README*",
                "CHANGELOG*",
                "HISTORY*",
                "LICENSE*",
                "COPYING*",
                "AUTHORS*",
                "CONTRIBUTORS*",
                "*.md",
                "*.markdown",
                "*.rst",
                "*.txt",
                # Git/Version control
                ".gitignore",
                ".gitattributes",
                ".gitmodules",
                ".hgignore",
                # CI/CD
                "*.yml",
                "*.yaml",
                ".github/**/*",
                ".gitlab-ci.yml",
                "azure-pipelines.yml",
                "Jenkinsfile",
                # Environment/Config
                ".env*",
                "*.ini",
                "*.cfg",
                "*.conf",
                "*.config",
                "*.toml",
                "*.properties",
                "*.prefs",
                "*.inf",
                "*.reg",
                # Docker/Container
                "Dockerfile*",
                "docker-compose*.yml",
                "docker-compose*.yaml",
                ".dockerignore",
                # IDE/Editor
                ".vscode/**/*",
                ".idea/**/*",
                "*.sublime-project",
                "*.sublime-workspace",
                ".editorconfig",
                # License/Legal
                "LICENSE",
                "LICENSE.*",
                "COPYING",
                "COPYING.*",
                "NOTICE",
                "NOTICE.*",
                # Misc important
                "*.manifest",
                "*.spec",
                "MANIFEST.in",
                "tox.ini",
                "pytest.ini",
                ".flake8",
                ".pylintrc",
                "mypy.ini",
                ".pre-commit-config.yaml",
                ".editorconfig",
                ".gitkeep",
                ".gitattributes",
                ".mailmap",
                "CODEOWNERS",
            ]
        )

    if args.keep_visualizations:
        # Keep visualization files
        exclude_file_patterns.extend(
            [
                "*.png",
                "*.jpg",
                "*.jpeg",
                "*.gif",
                "*.svg",
                "*.html",
            ]
        )

    if args.keep_reports:
        # Keep report files
        exclude_file_patterns.extend(
            [
                "*.json",
                "*.html",
                "*.pdf",
            ]
        )

    if not args.yes and not dry_run:
        print(f"About to clean temporary files under: {root_directory}")
        resp = input("Proceed? [y/N]: ").strip().lower()
        if resp not in ("y", "yes"):
            print("Aborted by user.")
            return 1

    if verbose:
        print("Starting cleanup process...")

    venv_names = (
        args.venv_names
        if args.venv_names is not None
        else (".venv", "venv", ".env", "env")
    )
    stats = delete_temporary_items(
        root_directory,
        dry_run=dry_run,
        verbose=verbose,
        include_dir_patterns=include_dir_patterns,
        include_file_patterns=include_file_patterns,
        exclude_dir_patterns=exclude_dir_patterns,
        exclude_file_patterns=exclude_file_patterns,
        remove_node_modules=args.remove_node_modules,
        remove_venvs=args.remove_venvs,
        venv_names=venv_names,
        follow_links=args.follow_links,
        max_depth=args.max_depth,
    )
    clear_log_files(
        root_directory,
        delete_logs_dir=args.delete_logs,
        dry_run=dry_run,
        verbose=verbose,
    )

    if args.prune_empty_dirs:
        prune_empty_dirs(root_directory, dry_run=dry_run, verbose=verbose)

    if verbose:
        print("\nCleanup completed")
        print(f"Directories removed: {stats['dirs_removed']}")
        print(f"Files removed: {stats['files_removed']}")
        if stats["errors"] > 0:
            print(f"Errors encountered: {stats['errors']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
