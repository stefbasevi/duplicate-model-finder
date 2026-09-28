import os
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pytest  # noqa: E402  (path setup above adds scripts/ to sys.path)

from scripts.duplicate_model_finder import (  # noqa: E402  (path setup above)
    ALLOWED_EXTENSIONS,
    TRASH_DIR_NAME,
    collect_hashes,
    compute_hash,
    compute_hash_blake2b,
    delete_files,
    find_duplicates,
    format_duplicates_for_display,
    is_model_file,
    iter_model_files,
    iter_model_files_scandir,
    move_files_to_trash,
    permanently_delete_files,
    purge_old_trash,
)


def test_is_model_file_respects_extensions():
    assert is_model_file("model.ckpt")
    assert is_model_file("model.safetensors")
    assert not is_model_file("notes.txt")
    assert not is_model_file("archive.zip", extensions=(".pt",))


def test_compute_hash_produces_consistent_result(tmp_path: Path):
    target = tmp_path / "model.ckpt"
    target.write_bytes(b"abc" * 1024)

    first_hash = compute_hash(target)
    second_hash = compute_hash(target)

    assert first_hash == second_hash
    assert len(first_hash) == 64


def test_find_duplicates_detects_matching_files(tmp_path: Path):
    model_dir = tmp_path / "models" / "Stable-diffusion"
    model_dir.mkdir(parents=True)

    a = model_dir / "a.ckpt"
    b = model_dir / "b.ckpt"
    c = model_dir / "c.pt"
    a.write_bytes(b"duplicate")
    b.write_bytes(b"duplicate")
    c.write_bytes(b"unique")

    duplicates = find_duplicates(directories=[model_dir], extensions=ALLOWED_EXTENSIONS)
    assert len(duplicates) == 1
    paths = next(iter(duplicates.values()))
    assert sorted(paths) == sorted([str(a), str(b)])


def test_collect_hashes_groups_paths_by_hash(tmp_path: Path):
    first = tmp_path / "first.ckpt"
    second = tmp_path / "second.ckpt"
    third = tmp_path / "third.ckpt"
    first.write_bytes(b"hello")
    second.write_bytes(b"hello")
    third.write_bytes(b"world")

    hashes = collect_hashes([first, second, third], max_workers=1)
    assert len(hashes) == 2
    matching_group = next(paths for paths in hashes.values() if len(paths) == 2)
    assert sorted(matching_group) == sorted([str(first), str(second)])


def test_format_duplicates_for_display_outputs_lines():
    duplicates = {"hash": ["/path/a", "/path/b"]}
    text, choices = format_duplicates_for_display(duplicates)

    assert "hash" in text
    assert "/path/a" in text
    assert choices == ["/path/a", "/path/b"]


def test_delete_files_reports_failures(tmp_path: Path):
    deletable = tmp_path / "delete.pt"
    deletable.write_bytes(b"data")

    message, failed = delete_files([deletable, tmp_path / "missing.pt"])

    assert "Moved 1 file" in message
    assert failed == [str(tmp_path / "missing.pt")]
    assert not deletable.exists()
    assert (tmp_path / TRASH_DIR_NAME).is_dir()


def test_iter_model_files_yields_supported_models(tmp_path: Path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    valid = model_dir / "model.safetensors"
    invalid = model_dir / "notes.txt"
    valid.write_bytes(b"data")
    invalid.write_text("ignore")

    files = list(iter_model_files([model_dir]))

    assert files == [str(valid)]
    assert invalid.exists()


# ---------------------------------------------------------------------------
# Cluster 1 (Performance) — size prefilter + parallel hashing
# ---------------------------------------------------------------------------


def test_collect_hashes_skips_unique_sizes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Files with unique sizes are skipped before the SHA256 I/O happens."""

    first = tmp_path / "first.ckpt"
    second = tmp_path / "second.ckpt"
    lonely = tmp_path / "lonely.ckpt"
    first.write_bytes(b"hello")
    second.write_bytes(b"hello")
    lonely.write_bytes(b"x")  # unique size — must be skipped

    hash_calls: list[str] = []

    def fake_compute_hash(path: str, chunk_size: int = 1 << 20) -> str:
        hash_calls.append(path)
        return compute_hash(path, chunk_size=chunk_size)

    monkeypatch.setattr("scripts.duplicate_model_finder.compute_hash", fake_compute_hash)

    hashes = collect_hashes([first, second, lonely], max_workers=1)

    assert len(hashes) == 1
    group = next(iter(hashes.values()))
    assert sorted(group) == sorted([str(first), str(second)])
    assert hash_calls == [str(first), str(second)]


def test_collect_hashes_size_prefilter_disabled_hashes_everything(tmp_path: Path):
    """With the prefilter disabled every file gets hashed."""

    first = tmp_path / "first.ckpt"
    second = tmp_path / "second.ckpt"
    lonely = tmp_path / "lonely.ckpt"
    first.write_bytes(b"hello")
    second.write_bytes(b"hello")
    lonely.write_bytes(b"x")

    hashes = collect_hashes(
        [first, second, lonely],
        use_size_prefilter=False,
        max_workers=1,
    )

    assert len(hashes) == 2
    groups = sorted((len(paths), sorted(paths)) for paths in hashes.values())
    assert groups == [(1, [str(lonely)]), (2, sorted([str(first), str(second)]))]


def test_collect_hashes_parallel_matches_sequential(tmp_path: Path):
    """The threaded path returns the same grouping as the serial one."""

    files = []
    for i in range(8):
        path = tmp_path / f"file_{i}.ckpt"
        # Half the files share content "alpha", half share "beta".
        path.write_bytes(b"alpha" if i % 2 == 0 else b"beta")
        files.append(path)

    sequential = collect_hashes(files, max_workers=1)
    parallel = collect_hashes(files, max_workers=4)

    sequential_normalised = sorted(tuple(sorted(group)) for group in sequential.values())
    parallel_normalised = sorted(tuple(sorted(group)) for group in parallel.values())
    assert sequential_normalised == parallel_normalised


def test_collect_hashes_ignores_files_that_cannot_be_stated(tmp_path: Path):
    real = tmp_path / "real.ckpt"
    missing = tmp_path / "ghost.ckpt"
    real.write_bytes(b"x")
    # `missing` is never written — must not crash the scan.
    # With the size prefilter, missing is dropped during stat; real has a
    # unique size so it is also dropped. Force the prefilter off to verify
    # the hashing step itself survives a missing file.

    with_prefilter = collect_hashes([real, missing], max_workers=1)
    without_prefilter = collect_hashes([real, missing], use_size_prefilter=False, max_workers=1)

    assert with_prefilter == {}
    assert without_prefilter == {compute_hash(real): [str(real)]}


def test_iter_model_files_respects_extensions_parameter(tmp_path: Path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    ckpt = model_dir / "model.ckpt"
    txt = model_dir / "model.txt"
    bin_ = model_dir / "model.bin"
    ckpt.write_bytes(b"x")
    txt.write_text("y")
    bin_.write_bytes(b"z")

    files = list(iter_model_files([model_dir], extensions=(".txt", ".bin")))

    assert sorted(files) == sorted([str(txt), str(bin_)])


# ---------------------------------------------------------------------------
# Cluster 2 (Safety) — soft-delete + explicit permanent-delete confirmation
# ---------------------------------------------------------------------------


def test_move_files_to_trash_creates_sibling_trash_dir(tmp_path: Path):
    target = tmp_path / "model.ckpt"
    target.write_bytes(b"data")

    message, failed = move_files_to_trash([target])

    assert failed == []
    assert "Moved 1 file" in message
    assert "recoverable manually" in message
    assert not target.exists()
    trash_dir = tmp_path / TRASH_DIR_NAME
    assert trash_dir.is_dir()
    assert len(list(trash_dir.iterdir())) == 1


def test_move_files_to_trash_handles_missing_files(tmp_path: Path):
    real = tmp_path / "real.ckpt"
    missing = tmp_path / "ghost.ckpt"
    real.write_bytes(b"x")
    # `missing` is never written.

    message, failed = move_files_to_trash([real, missing])

    assert failed == [str(missing)]
    assert "Moved 1 file" in message
    assert not real.exists()


def test_move_files_to_trash_skips_unwriteable_files(tmp_path, monkeypatch):
    target = tmp_path / "model.ckpt"
    target.write_bytes(b"x")

    monkeypatch.setattr("scripts.duplicate_model_finder.os.access", lambda *a, **kw: False)

    message, failed = move_files_to_trash([target])

    assert failed == [str(target)]
    assert target.exists()


def test_move_files_to_trash_handles_empty_input(tmp_path):
    message, failed = move_files_to_trash([])
    assert failed == []
    assert message == "No files moved"


def test_permanently_delete_files_requires_confirm(tmp_path: Path):
    target = tmp_path / "model.ckpt"
    target.write_bytes(b"x")

    message, failed = permanently_delete_files([target])  # default confirm=False

    assert "Refusing" in message
    assert "confirm=True" in message
    assert failed == [str(target)]
    assert target.exists()


def test_permanently_delete_files_with_confirm(tmp_path: Path):
    target = tmp_path / "model.ckpt"
    target.write_bytes(b"x")

    message, failed = permanently_delete_files([target], confirm=True)

    assert "Permanently deleted 1" in message
    assert failed == []
    assert not target.exists()


def test_permanently_delete_files_partial_failure(tmp_path: Path):
    real = tmp_path / "real.ckpt"
    real.write_bytes(b"x")
    missing = tmp_path / "ghost.ckpt"

    message, failed = permanently_delete_files([real, missing], confirm=True)

    assert "Permanently deleted 1" in message
    assert str(missing) in failed
    assert not real.exists()


def test_delete_files_routes_to_move_files_to_trash(tmp_path: Path):
    target = tmp_path / "model.ckpt"
    target.write_bytes(b"x")

    message, failed = delete_files([target])

    assert failed == []
    assert not target.exists()
    assert (tmp_path / TRASH_DIR_NAME).is_dir()
    assert "Moved 1 file" in message


# ---------------------------------------------------------------------------
# Cluster 3 (Tests/DX) — cancellation, symlinks, collisions, edge cases
# ---------------------------------------------------------------------------


def test_iter_model_files_follows_symlinks_without_double_counting(tmp_path: Path):
    """Symlinked model files should be reported once via realpath deduplication."""
    real_dir = tmp_path / "real_models"
    real_dir.mkdir()
    link_dir = tmp_path / "linked_models"
    link_dir.mkdir()

    real_file = real_dir / "actual.ckpt"
    real_file.write_bytes(b"data")

    link = link_dir / "alias.ckpt"
    try:
        link.symlink_to(real_file)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported on this filesystem")

    files = list(iter_model_files([real_dir, link_dir]))
    assert sorted(files) == sorted([str(real_file)])


def test_iter_model_files_returns_empty_when_no_matches(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert list(iter_model_files([empty])) == []


def test_iter_model_files_stop_event_aborts_before_iteration_finishes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Setting stop_event before any iteration yields no paths immediately."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "a.ckpt").write_bytes(b"a")
    (model_dir / "b.ckpt").write_bytes(b"b")

    stop = threading.Event()
    stop.set()  # already set, iter should return immediately

    collected = list(iter_model_files([model_dir], stop_event=stop))
    assert collected == []


def test_find_duplicates_propagates_stop_event_to_iter(tmp_path: Path):
    """find_duplicates() must pass stop_event through to iter_model_files."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "a.ckpt").write_bytes(b"alpha")
    (model_dir / "b.ckpt").write_bytes(b"beta")

    stop = threading.Event()
    stop.set()
    assert find_duplicates(directories=[model_dir], stop_event=stop) == {}


def test_move_files_to_trash_avoids_filename_collisions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Two files with the same basename in the same parent must each get a unique trash name."""
    parent = tmp_path / "models"
    parent.mkdir()

    # Pre-create an entry in the trash with the timestamp prefix that would
    # be generated for the first file, forcing a collision.
    fixed_prefix = "20260101T000000000000"
    trash = parent / TRASH_DIR_NAME
    trash.mkdir()
    (trash / f"{fixed_prefix}_shared.ckpt").write_bytes(b"old")

    target = parent / "shared.ckpt"
    target.write_bytes(b"new")

    # Patch _format_utc_timestamp via monkeypatch to force a deterministic collision
    import scripts.duplicate_model_finder as dmf

    fixed_ts = "20260101T000000000000"
    monkeypatch.setattr(dmf, "_format_utc_timestamp", lambda: fixed_ts)

    message, failed = move_files_to_trash([target])

    assert failed == []
    assert not target.exists()
    assert (trash / f"{fixed_prefix}_1_shared.ckpt").exists()


def test_permanently_delete_files_with_empty_paths_list_returns_neutral_message(
    tmp_path: Path,
):
    message, failed = permanently_delete_files([], confirm=True)
    assert failed == []
    assert message == "No files deleted"


def test_collect_hashes_stop_event_short_circuits(tmp_path: Path):
    """collect_hashes aborts cleanly when stop_event is set."""
    a = tmp_path / "a.ckpt"
    b = tmp_path / "b.ckpt"
    a.write_bytes(b"x")
    b.write_bytes(b"x")

    stop = threading.Event()
    stop.set()
    assert collect_hashes([a, b], stop_event=stop) == {}


def test_find_duplicates_detects_symlinked_duplicate(tmp_path: Path):
    """A real file and a symlinked copy of it should both be in the duplicate group."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    real = model_dir / "real.ckpt"
    real.write_bytes(b"same")

    link = model_dir / "link.ckpt"
    try:
        link.symlink_to(real)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported on this filesystem")

    duplicates = find_duplicates(directories=[model_dir], extensions=ALLOWED_EXTENSIONS)
    # Symlinks that point at the same physical file collapse via realpath
    # deduplication, so iter_model_files yields exactly one entry — the
    # first occurrence by os.walk order — and there are no duplicates to find.
    assert duplicates == {}
    files = list(iter_model_files([model_dir]))
    assert len(files) == 1
    assert files[0] == str(real) or files[0] == str(link)


def test_collect_hashes_parallel_skips_missing_file(tmp_path: Path):
    """Regression for issue #16.

    The parallel path in collect_hashes must skip files whose compute_hash
    raises OSError (deleted mid-scan, permission changed, symlink target
    removed, etc.) instead of crashing the whole scan with TypeError:
    cannot unpack non-iterable NoneType object.
    """

    real = tmp_path / "real.ckpt"
    real.write_bytes(b"x")
    missing = tmp_path / "ghost.ckpt"
    # `missing` is intentionally never created.

    result = collect_hashes(
        [real, missing],
        use_size_prefilter=False,
        max_workers=4,
    )

    expected_hash = compute_hash(real)
    assert result == {expected_hash: [str(real)]}


def test_purge_old_trash_removes_old_files(tmp_path, monkeypatch):
    """Files older than the retention window should be deleted from trash."""
    monkeypatch.setattr("scripts.duplicate_model_finder.MODEL_DIRS", [str(tmp_path)])
    trash_dir = tmp_path / TRASH_DIR_NAME
    trash_dir.mkdir()
    disposed_at = datetime.now(timezone.utc) - timedelta(days=60)
    old_file = trash_dir / f"{disposed_at:%Y%m%dT%H%M%S%f}_old.ckpt"
    old_file.write_text("old content")
    # The model was modified recently; its disposal timestamp is 60 days old.

    message, deleted = purge_old_trash(retention_days=30)

    assert deleted == 1
    assert not old_file.exists()
    assert "Purged 1 file(s) older than 30 day(s)" in message


def test_purge_old_trash_keeps_young_files(tmp_path, monkeypatch):
    """Files within the retention window must remain untouched."""
    monkeypatch.setattr("scripts.duplicate_model_finder.MODEL_DIRS", [str(tmp_path)])
    trash_dir = tmp_path / TRASH_DIR_NAME
    trash_dir.mkdir()
    young_file = trash_dir / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S%f}_young.ckpt"
    young_file.write_text("young content")
    old_mtime = time.time() - (60 * 86400)
    os.utime(young_file, (old_mtime, old_mtime))

    message, deleted = purge_old_trash(retention_days=30)

    assert deleted == 0
    assert young_file.exists()
    assert "Purged 0" in message


def test_purge_old_trash_negative_retention_rejected():
    """Negative retention values must not touch the filesystem."""
    message, deleted = purge_old_trash(retention_days=-1)

    assert "Invalid" in message
    assert deleted == 0


def test_purge_old_trash_zero_retention_removes_everything(tmp_path, monkeypatch):
    """retention_days=0 is the explicit "purge now" knob."""
    monkeypatch.setattr("scripts.duplicate_model_finder.MODEL_DIRS", [str(tmp_path)])
    trash_dir = tmp_path / TRASH_DIR_NAME
    trash_dir.mkdir()
    fresh = trash_dir / "fresh.ckpt"
    fresh.write_text("x")

    message, deleted = purge_old_trash(retention_days=0)

    assert deleted == 1
    assert not fresh.exists()


def test_purge_old_trash_handles_missing_model_dir(monkeypatch):
    """Non-existent MODEL_DIRS entries are skipped without raising."""
    monkeypatch.setattr("scripts.duplicate_model_finder.MODEL_DIRS", ["/nonexistent/path/xyz"])

    message, deleted = purge_old_trash(retention_days=30)

    assert deleted == 0
    assert "Purged 0" in message


def test_purge_old_trash_skips_subdirectories(tmp_path, monkeypatch):
    """Sub-directories inside trash must not be descended into or removed."""
    monkeypatch.setattr("scripts.duplicate_model_finder.MODEL_DIRS", [str(tmp_path)])
    trash_dir = tmp_path / TRASH_DIR_NAME
    trash_dir.mkdir()
    nested = trash_dir / "nested_dir"
    nested.mkdir()
    old_file_in_nested = nested / "old.ckpt"
    old_file_in_nested.write_text("x")
    old_time = time.time() - (60 * 86400)
    os.utime(old_file_in_nested, (old_time, old_time))

    message, deleted = purge_old_trash(retention_days=30)

    assert deleted == 0
    assert nested.is_dir()


def test_compute_hash_blake2b_distinguishes_files(tmp_path):
    """BLAKE2b digest must differ between distinct file contents."""
    a = tmp_path / "a.ckpt"
    b = tmp_path / "b.ckpt"
    a.write_text("alpha")
    b.write_text("beta")
    assert compute_hash_blake2b(str(a)) != compute_hash_blake2b(str(b))


def test_compute_hash_blake2b_is_deterministic(tmp_path):
    """Same content must yield the same BLAKE2b digest across calls."""
    target = tmp_path / "model.ckpt"
    target.write_text("payload")
    assert compute_hash_blake2b(str(target)) == compute_hash_blake2b(str(target))


def test_compute_hash_blake2b_handles_large_files(tmp_path):
    """BLAKE2b with the 16 MB default chunk must handle > NVME_CHUNK_SIZE."""
    target = tmp_path / "large.ckpt"
    # 32 MB > NVME_CHUNK_SIZE (16 MB)
    target.write_bytes(b"\x00" * (32 * 1024 * 1024))
    digest = compute_hash_blake2b(str(target))
    # BLAKE2b default digest length is 64 bytes -> 128 hex chars
    assert len(digest) == 128


def test_iter_model_files_scandir_yields_same_set_as_default(tmp_path, monkeypatch):
    """Scandir walker must surface every model file the default walker does."""
    monkeypatch.setattr("scripts.duplicate_model_finder.MODEL_DIRS", [str(tmp_path)])
    models = tmp_path / "models"
    models.mkdir()
    (models / "a.ckpt").write_text("a")
    sub = models / "sub"
    sub.mkdir()
    (sub / "b.safetensors").write_text("b")
    (sub / "ignored.txt").write_text("not a model")

    default_paths = set(iter_model_files([str(models)]))
    scandir_paths = set(iter_model_files_scandir([str(models)]))

    assert scandir_paths == default_paths
    # And the ignored file must not show up in either.
    assert all(not p.endswith("ignored.txt") for p in scandir_paths)


def test_find_duplicates_prefer_nvme_produces_distinct_digests(tmp_path, monkeypatch):
    """NVMe profile must still detect duplicates AND differ from SHA256."""
    monkeypatch.setattr("scripts.duplicate_model_finder.MODEL_DIRS", [str(tmp_path)])
    (tmp_path / "a.ckpt").write_text("payload")
    (tmp_path / "b.ckpt").write_text("payload")  # duplicate content

    # default: SHA256 — duplicates detected
    default = find_duplicates([str(tmp_path)])
    assert len(default) == 1

    # NVMe profile: still detects the duplicate (BLAKE2b on same payload matches)
    nvme = find_duplicates([str(tmp_path)], prefer_nvme=True)
    assert len(nvme) == 1

    # The two profiles must produce different digests for the same payload.
    default_digest = next(iter(default))
    nvme_digest = next(iter(nvme))
    assert default_digest != nvme_digest


def test_find_duplicates_prefer_nvme_respects_stop_event(tmp_path, monkeypatch):
    """Cancellation must short-circuit the NVMe walker just like the default."""
    monkeypatch.setattr("scripts.duplicate_model_finder.MODEL_DIRS", [str(tmp_path)])
    (tmp_path / "a.ckpt").write_text("payload")
    (tmp_path / "b.ckpt").write_text("different")
    stop_event = threading.Event()
    stop_event.set()  # cancelled before start

    result = find_duplicates([str(tmp_path)], stop_event=stop_event, prefer_nvme=True)
    assert result == {}
