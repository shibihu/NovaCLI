#!/usr/bin/env python3
"""Manual test for Windows PTY implementation."""

from nova.core.pty import PTYManager, get_pty_backend_info
import time

print("=== PTY Backend Information ===")
info = get_pty_backend_info()
for key, value in info.items():
    print(f"{key:20} {value}")

print("\n=== Creating PTY Session ===")
manager = PTYManager()
session = manager.create('.')

print(f"Session ID:     {session.id}")
print(f"Platform:       {session.__class__.__module__}.{session.__class__.__name__}")
print(f"Is alive:       {session.is_alive}")
print(f"PID:            {session.pid}")
print(f"Working dir:    {session.cwd}")
print(f"Terminal size:  {session.cols}x{session.rows}")

print("\n=== Testing I/O ===")
session.write('echo Hello from Windows PTY!\r\n')
time.sleep(0.5)

output = b""
for _ in range(5):
    chunk = session.read(1024)
    if chunk:
        output += chunk
    time.sleep(0.1)

if output:
    decoded = output.decode("utf-8", errors="replace")
    print(f"Output received ({len(output)} bytes):")
    print(decoded[:200])
else:
    print("No output received (expected for buffered shell)")

print("\n=== Testing Resize ===")
session.resize(100, 30)
print(f"Terminal size:  {session.cols}x{session.rows}")

print("\n=== Closing Session ===")
session.close()
print(f"Is alive:       {session.is_alive}")
print("Session closed successfully")
