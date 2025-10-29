"""
Patched version of stream_pipeline_online.py with better close() handling.
This adds timeouts and diagnostics to prevent infinite hangs.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from stream_pipeline_offline (used for both offline and online modes)
from stream_pipeline_offline import *
import time

# Monkey-patch the StreamSDK.close method
original_close = StreamSDK.close

def close_with_timeout(self, timeout=60):
    """
    Enhanced close() with timeout and diagnostics.
    """
    print(f"\n[CLOSE] Signaling audio2motion_worker to stop...")
    self.audio2motion_queue.put(None)

    thread_names = [
        "audio2motion_worker",
        "motion_stitch_worker",
        "warp_f3d_worker",
        "decode_f3d_worker",
        "putback_worker",
        "writer_worker",
    ]

    print(f"[CLOSE] Waiting for {len(self.thread_list)} worker threads to finish...")
    print(f"[CLOSE] Timeout: {timeout}s")

    start_time = time.time()

    for i, thread in enumerate(self.thread_list):
        thread_name = thread_names[i] if i < len(thread_names) else f"thread_{i}"
        remaining_timeout = timeout - (time.time() - start_time)

        if remaining_timeout <= 0:
            print(f"[CLOSE] ✗ Timeout reached!")
            print(f"[CLOSE] Still waiting for: {thread_name}")
            break

        print(f"[CLOSE] Waiting for {thread_name}... ", end="", flush=True)
        thread.join(timeout=remaining_timeout)

        if thread.is_alive():
            print(f"✗ TIMEOUT ({remaining_timeout:.1f}s)")
            print(f"[CLOSE] Thread {thread_name} is stuck!")

            # Show queue sizes for debugging
            print(f"[CLOSE] Queue sizes:")
            print(f"  audio2motion_queue: {self.audio2motion_queue.qsize()}")
            print(f"  motion_stitch_queue: {self.motion_stitch_queue.qsize()}")
            print(f"  warp_f3d_queue: {self.warp_f3d_queue.qsize()}")
            print(f"  decode_f3d_queue: {self.decode_f3d_queue.qsize()}")
            print(f"  putback_queue: {self.putback_queue.qsize()}")
            print(f"  writer_queue: {self.writer_queue.qsize()}")

            # Force stop event
            print(f"[CLOSE] Setting stop_event to force threads to exit...")
            self.stop_event.set()

            # Give threads 5 more seconds to notice stop_event
            thread.join(timeout=5)

            if thread.is_alive():
                print(f"[CLOSE] ✗ Thread still alive after stop_event!")
                print(f"[CLOSE] This thread is deadlocked or blocking on I/O")
            else:
                print(f"[CLOSE] ✓ Thread stopped after stop_event")
        else:
            elapsed = time.time() - start_time
            print(f"✓ ({elapsed:.2f}s)")

    # Check if all threads finished
    alive_threads = [i for i, t in enumerate(self.thread_list) if t.is_alive()]
    if alive_threads:
        print(f"\n[CLOSE] ✗ Warning: {len(alive_threads)} thread(s) still alive:")
        for i in alive_threads:
            name = thread_names[i] if i < len(thread_names) else f"thread_{i}"
            print(f"  - {name}")
    else:
        print(f"\n[CLOSE] ✓ All threads finished successfully")

    try:
        self.writer.close()
        self.writer_pbar.close()
    except:
        traceback.print_exc()

    # Check if any worker encountered an exception
    if self.worker_exception is not None:
        print(f"[CLOSE] ✗ Worker exception: {self.worker_exception}")
        raise self.worker_exception

# Apply the patch
StreamSDK.close = close_with_timeout

# Export everything from the original module
__all__ = ['StreamSDK']
