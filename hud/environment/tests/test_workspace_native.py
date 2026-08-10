"""Native Workspace SSH file-write contracts."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from typing import TYPE_CHECKING

import asyncssh
import pytest

from hud.capabilities import SSHClient
from hud.environment.workspace import Workspace

if TYPE_CHECKING:
    from pathlib import Path


async def _connect(ws: Workspace) -> asyncssh.SSHClientConnection:
    host, port = ws.ssh_url.removeprefix("ssh://").split(":")
    key_path = ws.ssh_client_key_path
    assert key_path is not None
    return await asyncssh.connect(
        host,
        int(port),
        username=ws.ssh_user,
        client_keys=[str(key_path)],
        known_hosts=None,
    )


def _wait_for_path(path: Path) -> None:
    deadline = time.monotonic() + 5
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {path}")
        time.sleep(0.01)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell interruption")
async def test_cancelled_file_write_keeps_existing_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    bin_dir = root / "bin"
    bin_dir.mkdir(parents=True)
    blocking_cat = bin_dir / "cat"
    blocking_cat.write_text(
        "#!/bin/sh\n/bin/cat\n: > .write-staged\nsleep 60\n",
        encoding="utf-8",
    )
    blocking_cat.chmod(0o755)
    destination = root / "destination.txt"
    destination.write_text("old", encoding="utf-8")
    ws = Workspace(root)
    await ws.start()
    try:
        async with await _connect(ws) as conn:
            client = SSHClient(ws.capability(), conn)
            original_run = client.run

            async def run(
                command: object, *args: object, **kwargs: object
            ) -> asyncssh.SSHCompletedProcess:
                assert isinstance(command, str)
                command = command.replace("cat >", f"{blocking_cat} >", 1)
                return await original_run(command, *args, **kwargs)

            monkeypatch.setattr(client, "run", run)
            write = asyncio.create_task(client.write_text("destination.txt", "new"))
            await asyncio.to_thread(_wait_for_path, root / ".write-staged")

            assert destination.read_text(encoding="utf-8") == "old"
            write.cancel()
            with pytest.raises(asyncio.CancelledError):
                await write
    finally:
        await ws.stop()

    assert destination.read_text(encoding="utf-8") == "old"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell commit control")
async def test_file_write_atomically_replaces_symlink_target_with_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    bin_dir = root / "bin"
    bin_dir.mkdir(parents=True)
    blocking_mv = bin_dir / "mv"
    blocking_mv.write_text(
        "#!/bin/sh\n: > .commit-ready\n"
        'while [ ! -e .allow-commit ];do sleep 0.01;done\n/bin/mv "$@"\n',
        encoding="utf-8",
    )
    blocking_mv.chmod(0o755)
    target = root / "target.txt"
    target.write_text("old", encoding="utf-8")
    target.chmod(0o640)
    metadata = (target.stat().st_mode, target.stat().st_uid, target.stat().st_gid)
    link = root / "link.txt"
    link.symlink_to(target.name)
    ws = Workspace(root)
    await ws.start()
    try:
        async with await _connect(ws) as conn:
            client = SSHClient(ws.capability(), conn)
            original_run = client.run

            async def run(
                command: object, *args: object, **kwargs: object
            ) -> asyncssh.SSHCompletedProcess:
                assert isinstance(command, str)
                command = command.replace("mv -f --", f"{blocking_mv} -f --", 1)
                return await original_run(command, *args, **kwargs)

            monkeypatch.setattr(client, "run", run)
            write = asyncio.create_task(client.write_text("link.txt", "complete new content"))
            await asyncio.to_thread(_wait_for_path, root / ".commit-ready")

            assert target.read_text(encoding="utf-8") == "old"
            assert link.is_symlink()
            staged = list(root.glob(".hud-write.*"))
            assert len(staged) == 1
            assert staged[0].read_text(encoding="utf-8") == "complete new content"

            (root / ".allow-commit").touch()
            await write
    finally:
        await ws.stop()

    assert target.read_text(encoding="utf-8") == "complete new content"
    assert link.is_symlink()
    assert (target.stat().st_mode, target.stat().st_uid, target.stat().st_gid) == metadata
    assert not list(root.glob(".hud-write.*"))


def _powershell(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        check=True,
        capture_output=True,
        text=True,
    )


def _quoted(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows file replacement contract")
async def test_windows_file_write_replaces_symlink_target_with_security_metadata(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = root / "target.txt"
    target.write_text("old", encoding="utf-8")
    link = root / "link.txt"
    link.symlink_to(target.name)
    await asyncio.to_thread(
        _powershell,
        (
            f"$a=[IO.File]::GetAccessControl({_quoted(target)});"
            "$a.SetOwner([Security.Principal.WindowsIdentity]::GetCurrent().User);"
            "$r=[Security.AccessControl.FileSystemAccessRule]::new("
            "[Security.Principal.SecurityIdentifier]::new('S-1-1-0'),"
            "[Security.AccessControl.FileSystemRights]::ReadData,"
            "[Security.AccessControl.AccessControlType]::Allow);"
            "$a.AddAccessRule($r);"
            f"[IO.File]::SetAccessControl({_quoted(target)},$a)"
        ),
    )
    before = (
        await asyncio.to_thread(
            _powershell,
            f"[Console]::Out.Write([IO.File]::GetAccessControl({_quoted(target)}).Sddl)",
        )
    ).stdout
    content = "complete new content\n" * 1024
    ws = Workspace(root)
    await ws.start()
    try:
        async with await _connect(ws) as conn:
            await SSHClient(ws.capability(), conn).write_text("link.txt", content)
    finally:
        await ws.stop()

    after = (
        await asyncio.to_thread(
            _powershell,
            f"[Console]::Out.Write([IO.File]::GetAccessControl({_quoted(target)}).Sddl)",
        )
    ).stdout
    assert target.read_text(encoding="utf-8") == content
    assert link.is_symlink()
    assert after == before
    assert not list(root.glob(".hud-write-*"))


@pytest.mark.skipif(sys.platform != "win32", reason="Windows file replacement contract")
async def test_windows_failed_commit_keeps_existing_destination(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    destination = root / "destination.txt"
    destination.write_text("old", encoding="utf-8")
    lock = await asyncio.create_subprocess_exec(
        "powershell",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        (
            f"$f=[IO.File]::Open({_quoted(destination)},[IO.FileMode]::Open,"
            "[IO.FileAccess]::Read,[IO.FileShare]::Read);"
            "[Console]::Out.WriteLine('ready');[Console]::In.ReadLine();$f.Dispose()"
        ),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    assert lock.stdout is not None
    assert (await lock.stdout.readline()).strip() == b"ready"
    ws = Workspace(root)
    await ws.start()
    try:
        async with await _connect(ws) as conn:
            client = SSHClient(ws.capability(), conn)
            with pytest.raises(asyncssh.ProcessError):
                await client.write_text("destination.txt", "new" * 4096)
    finally:
        await ws.stop()
        assert lock.stdin is not None
        lock.stdin.write(b"\n")
        await lock.stdin.drain()
        lock.stdin.close()
        await asyncio.wait_for(lock.wait(), 5)

    assert destination.read_text(encoding="utf-8") == "old"
    assert not list(root.glob(".hud-write-*"))
