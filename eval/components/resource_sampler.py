import logging
import os
import threading
import time
from dataclasses import dataclass

import psutil


LOGGER = logging.getLogger("eval.resource_sampler")


@dataclass(frozen=True)
class ResourceSample:
    ts_mono: float
    cpu_total_pct: float
    rss_total_bytes: int


class ResourceSampler(threading.Thread):
    def __init__(self, root_pid: int | None = None, interval_s: float = 0.1):
        super().__init__(daemon=True)
        self.root_pid = root_pid if root_pid is not None else os.getpid()
        self.interval_s = interval_s

        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._samples: list[ResourceSample] = []
        self._known_procs: dict[int, psutil.Process] = {}
        self._last_cpu_time_total: float | None = None
        self._last_ts_mono: float | None = None

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        LOGGER.info(
            "Resource sampler started | root_pid=%s | interval_s=%.3f",
            self.root_pid,
            self.interval_s,
        )
        last_debug = time.monotonic()
        while not self._stop_event.is_set():
            now = time.monotonic()
            processes = self._discover_processes()

            total_cpu_time = 0.0
            total_rss = 0

            for proc in processes:
                try:
                    if proc.pid not in self._known_procs:
                        self._known_procs[proc.pid] = proc

                    cpu_times = proc.cpu_times()
                    total_cpu_time += cpu_times.user + cpu_times.system
                    total_rss += proc.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue

            cpu_pct = 0.0
            if self._last_cpu_time_total is not None and self._last_ts_mono is not None:
                delta_proc = max(0.0, total_cpu_time - self._last_cpu_time_total)
                delta_wall = max(1e-9, now - self._last_ts_mono)
                # Aggregate process CPU utilization; can exceed 100% on multicore workloads.
                cpu_pct = (delta_proc / delta_wall) * 100.0

            self._last_cpu_time_total = total_cpu_time
            self._last_ts_mono = now

            sample = ResourceSample(
                ts_mono=now,
                cpu_total_pct=cpu_pct,
                rss_total_bytes=total_rss,
            )

            with self._lock:
                self._samples.append(sample)

            if now - last_debug >= 2.0:
                LOGGER.debug(
                    "Resource heartbeat | process_count=%s | cpu_total_pct=%.2f | rss_mb=%.2f",
                    len(processes),
                    cpu_pct,
                    total_rss / (1024 * 1024),
                )
                last_debug = now

            self._stop_event.wait(self.interval_s)

        LOGGER.info("Resource sampler stopped | samples=%s", len(self.get_samples()))

    def _discover_processes(self) -> list[psutil.Process]:
        try:
            root = psutil.Process(self.root_pid)
            children = root.children(recursive=True)
            return [root, *children]
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return []

    def get_samples(self) -> list[ResourceSample]:
        with self._lock:
            return list(self._samples)

    def mean_cpu_between(self, start_ts: float, end_ts: float) -> float:
        samples = [
            s.cpu_total_pct
            for s in self.get_samples()
            if start_ts <= s.ts_mono <= end_ts
        ]
        if not samples:
            return 0.0
        return sum(samples) / len(samples)

    def peak_memory_mb_between(self, start_ts: float, end_ts: float) -> float:
        samples = [
            s.rss_total_bytes
            for s in self.get_samples()
            if start_ts <= s.ts_mono <= end_ts
        ]
        if not samples:
            return 0.0
        return max(samples) / (1024 * 1024)
