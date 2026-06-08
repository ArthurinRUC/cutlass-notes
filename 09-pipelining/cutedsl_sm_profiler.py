"""Reusable, application-agnostic in-kernel SM profiler for CuTe DSL kernels.

A close CuTe DSL port of ``magnus_a2a``'s ``sm_profiler`` (CUDA C++). The
profiler is kernel-agnostic: callers register their own event *types*
(``event_no``) via :meth:`SmProfiler.register_event` and instrument code with
:func:`event_start` / :func:`event_end`.

Model
-----
Each (block, warp) appends events into one global ``int64`` buffer: a per-warp
counter hands out a monotonically increasing ``event_id``, so events land in
order. A range event stores a type (``event_no``) plus start/end timestamps
from ``%globaltimer``, each preceded by a compiler barrier so the read is not
reordered across the surrounding work. No atomics -- only lane 0 writes a given
slot. Overflow drops: past ``max_events_per_warp`` the counter keeps advancing
but no records are written.

Two deviations from the original cut per-event memory traffic: a :class:`Session`
(opened once via :func:`begin`) caches the header + this warp's ids in registers
instead of re-parsing per call; and ``event_start`` returns the ``event_id`` in a
register for ``event_end`` to reuse, replacing the original's global ``active``
table round-trip.

Buffer layout (single 1-D int64 tensor, element offsets;
``total_warps = num_blocks * num_warps``, ``E = max_events_per_warp``,
``F = NUM_EV_FIELDS``)::

    header   : [0, HEADER)                num_blocks, num_warps, max_events
    smid     : [HEADER, +total_warps)     sm_id per (block, warp), written once
    counters : [.., +total_warps)         one per (block, warp)
    events   : [.., +total_warps*E*F)     F fields per (block, warp, event_id)

(block, warp, event_id) are implicit in the index and ``sm_id`` is per-warp, so
each record stores only (event_no, st_ts, en_ts).
"""

from __future__ import annotations

import json
from collections import namedtuple
from dataclasses import dataclass, field

import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cute.runtime import from_dlpack


# ---------------------------------------------------------------------------
# Special-register reads + compiler barrier.
# ---------------------------------------------------------------------------
def _globaltimer_ns():
    """Read ``%globaltimer`` (ns) via a SIDE-EFFECTING inline asm.

    Not the pure ``nvvm.read_ptx_sreg_globaltimer`` intrinsic: two pure SREG
    reads in straight-line code get CSE'd into one, making an event's two stamps
    (e.g. ``EV_ST`` / ``EV_ST_LATE``) identical and collapsing the overhead band
    to zero. A ``volatile`` ``mov.u64`` cannot be CSE'd, so each call reads at its
    own program point.
    """
    return cutlass.Int64(
        llvm.inline_asm(
            ir.IntegerType.get_signless(64),
            [],
            "mov.u64 $0, %globaltimer;",
            "=l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


def _smid():
    return cutlass.Int64(nvvm.read_ptx_sreg_smid(ir.IntegerType.get_signless(32)))


def _compiler_barrier():
    """Empty ``asm volatile`` with a memory clobber -- a pure compiler barrier.

    Emits no instruction but stops the compiler from scheduling the adjacent
    ``%globaltimer`` read across the work it brackets; only constrains ordering,
    unlike a hardware ``membar``.
    """
    llvm.inline_asm(
        None,
        [],
        "",
        "~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


# ---------------------------------------------------------------------------
# Buffer schema (shared by device writers and host readers). No application
# event types live here -- callers register their own.
# ---------------------------------------------------------------------------
HDR_NUM_BLOCKS = 0
HDR_NUM_WARPS = 1
HDR_MAX_EVENTS = 2
HEADER = 3

# Per-event record fields (int64). ``EV_ST`` / ``EV_EN`` bound the event and are
# stretched to swallow the profiler's own bookkeeping: ``EV_ST`` is the FIRST
# instruction of ``event_start``, ``EV_EN`` the LAST of ``event_end``, so
# adjacent events abut instead of leaving an overhead gap.
#
# ``EV_ST_LATE`` / ``EV_EN_EARLY`` are written ONLY under ``show_overhead``; they
# mark the inner edges of the two overhead bands (``[EV_ST, EV_ST_LATE]`` =
# start-side cost, ``[EV_EN_EARLY, EV_EN]`` = end-side cost, with the inner
# ``[EV_ST_LATE, EV_EN_EARLY]`` the pure work). Zero (and ignored on readout)
# when off or for dropped events.
EV_EVENT_NO = 0
EV_ST = 1
EV_EN = 2
EV_ST_LATE = 3
EV_EN_EARLY = 4
NUM_EV_FIELDS = 5

# Each warp's append counter gets its own 128-byte L2 line (16 int64) instead of
# sitting in a packed ``counters[total_warps]`` array. Reason: the warps of one
# block run on the same SM and their counter slots would otherwise share a cache
# line, so the per-event counter read-modify-write false-shares and serializes
# across warps. Padding to a line-per-counter removes that contention. Only slot
# 0 of each stride is used; the padding bytes are never read.
COUNTER_STRIDE = 16

# One parsed range event. ``st_late`` / ``en_early`` are 0 unless the run had
# ``show_overhead`` on; the readout treats a 0 as "no band".
Record = namedtuple(
    "Record",
    "block warp event_id event_no sm_id st en st_late en_early",
)


# ---------------------------------------------------------------------------
# Device-side API (call from inside a @cute.kernel): open a session once with
# ``prof.begin()``, then ``session.event_start(no)`` / ``session.event_end(no)``.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Session:
    """Per-invocation profiler context (open with :meth:`ProfilerArgs.begin`).

    Holds this warp's coordinates, cached once by ``begin``. It is a
    DynamicExpression (implements the flatten protocol below) so it can be
    referenced inside a dynamic ``cutlass.range`` loop -- the DSL must flatten
    every value live in the loop body, which a plain Python object cannot do.
    Its only mutable state is the ``_open`` map, which lives purely at trace time
    (no runtime cost). When profiling is disabled the coordinate fields are
    ``None`` and every method folds to nothing.
    """

    prof: ProfilerArgs
    warp_global: object
    max_events: object
    counters_base: object
    events_base: object
    # Compile-time map ``event_no -> open event_id`` so callers can close an event
    # by its (constexpr) TYPE rather than threading the returned id. Resolved
    # entirely at trace time: it parks a register ``Value``, materializes nothing,
    # and adds no runtime traffic. Mutated in place (allowed on a frozen instance);
    # excluded from the flatten protocol and from eq/hash. The DSL hands the loop
    # body a freshly reconstructed Session (empty map), which is why same-type
    # events must open and close within the same trace region -- true for this
    # profiler's call pattern, and the same single-live-event-per-type limit the
    # original's ``active`` table had.
    _open: dict = field(default_factory=dict, compare=False)

    # --- DynamicExpression protocol: flatten to the buffer's MLIR values plus
    # the four cached coordinates; ``enabled`` stays a compile-time constant.
    def __extract_mlir_values__(self):
        if not self.prof.enabled:
            return []
        vals = list(self.prof.__extract_mlir_values__())
        for coord in (self.warp_global, self.max_events, self.counters_base, self.events_base):
            vals += list(coord.__extract_mlir_values__())
        return vals

    def __new_from_mlir_values__(self, values):
        if not self.prof.enabled:
            return self
        n = len(self.prof.__extract_mlir_values__())
        new_prof = self.prof.__new_from_mlir_values__(values[:n])
        rest = values[n:]
        coords = []
        for coord in (self.warp_global, self.max_events, self.counters_base, self.events_base):
            k = len(coord.__extract_mlir_values__())
            coords.append(coord.__new_from_mlir_values__(rest[:k]))
            rest = rest[k:]
        # Share the SAME _open dict across reconstructions: the DSL flattens +
        # reconstructs the session around any region (e.g. the prologue scf.for)
        # it stays live across, and a fresh default_factory dict would lose the
        # open-id parked before that region -- the type-keyed close would then
        # KeyError. Sharing keeps one map for the whole trace.
        return Session(new_prof, *coords, _open=self._open)

    def _rec(self, event_id):
        return self.events_base + (self.warp_global * self.max_events + event_id) * NUM_EV_FIELDS

    @cute.jit
    def _alloc(self, event_no: cutlass.Constexpr, start_ts):
        """Lane-0 helper: bump the counter, write the start record, return the id.

        Drops the record (counter still advances) once the warp is full. ``sm_id``
        is per-warp (written by begin) and ``EV_EN`` is left zeroed for the close.
        """
        counter_off = self.counters_base + self.warp_global * COUNTER_STRIDE
        event_id = self.prof.buffer[counter_off]
        self.prof.buffer[counter_off] = event_id + 1
        if event_id < self.max_events:
            rec = self._rec(event_id)
            self.prof.buffer[rec + EV_EVENT_NO] = cutlass.Int64(event_no)
            self.prof.buffer[rec + EV_ST] = start_ts
            if cutlass.const_expr(self.prof.show_overhead):
                # Latest start stamp: [EV_ST, EV_ST_LATE] = start-side overhead.
                _compiler_barrier()
                self.prof.buffer[rec + EV_ST_LATE] = _globaltimer_ns()
        return event_id

    @cute.jit
    def event_start(self, event_no: cutlass.Constexpr):
        """Open a range event of type ``event_no`` (close it with the SAME type).

        No-op when disabled. Lane 0 allocates the next event_id (counter RMW, no
        atomics) and writes the start record; the id is parked in the session's
        compile-time ``_open`` map so :meth:`event_end` can recover it from the
        type alone. The start stamp is the FIRST instruction so the interval
        swallows this call's bookkeeping; under ``show_overhead`` a second stamp
        (``EV_ST_LATE``) bounds the start-side overhead band.

        ``@cute.jit`` runs the AST preprocessor so the dynamic ``if`` below is
        plain Python (lowered to ``scf.if``); inlined into the kernel trace.
        """
        if cutlass.const_expr(self.prof.enabled):
            # Lane 0 is the sole writer; other lanes keep the sentinel -1. Seeded
            # before the branch because the DSL requires a value live past a
            # dynamic ``if`` to exist beforehand (the branch only updates it).
            event_id = cutlass.Int64(-1)
            if cute.arch.lane_idx() == 0:
                # Earliest start stamp: before the counter RMW and record writes.
                # The barrier stops the compiler from sinking it past them.
                _compiler_barrier()
                event_id = self._alloc(event_no, _globaltimer_ns())
            # Park the warp-wide id (the scf.if result) under its type, at trace
            # time -- nothing is emitted; event_end reads it back.
            self._open[event_no] = event_id
            return event_id
        return None

    @cute.jit
    def event_end(self, event_no: cutlass.Constexpr):
        """Close the open range of type ``event_no`` (opened by :meth:`event_start`).

        Takes the event TYPE, not a runtime id: the id is recovered from the
        compile-time ``_open`` map (a trace-time lookup, no global round-trip).
        No-op when disabled; skips an id that overflowed the per-warp capacity.
        ``EV_EN`` is the LAST stamp taken, so the interval reaches as far as it
        can toward the next event; under ``show_overhead`` an earlier stamp
        (``EV_EN_EARLY``) bounds the end-side band ``[EV_EN_EARLY, EV_EN]``.
        """
        if cutlass.const_expr(self.prof.enabled):
            event_id = self._open[event_no]
            # Lane 0 (the lane that opened the event) is the sole writer.
            if cute.arch.lane_idx() == 0:
                if event_id < self.max_events:
                    rec = self._rec(event_id)
                    if cutlass.const_expr(self.prof.show_overhead):
                        # Earliest end stamp; the later EV_EN below covers this
                        # store, so [EV_EN_EARLY, EV_EN] = end-side overhead.
                        _compiler_barrier()
                        self.prof.buffer[rec + EV_EN_EARLY] = _globaltimer_ns()
                    _compiler_barrier()
                    self.prof.buffer[rec + EV_EN] = _globaltimer_ns()

    @cute.jit
    def event_switch(self, close_no: cutlass.Constexpr, open_no: cutlass.Constexpr):
        """Close ``close_no`` and open ``open_no`` sharing ONE ``%globaltimer`` read.

        For a back-to-back boundary (``event_end`` immediately followed by
        ``event_start``) this is both cheaper -- one timer read instead of two --
        and more faithful: the closing ``EV_EN`` and opening ``EV_ST`` are the
        SAME stamp, so the two slices abut exactly with no profiler-overhead gap.
        Under ``show_overhead`` the switch's bookkeeping shows up as the opening
        event's start band (the closed event gets no end band).
        """
        if cutlass.const_expr(self.prof.enabled):
            open_id = cutlass.Int64(-1)
            if cute.arch.lane_idx() == 0:
                _compiler_barrier()
                boundary_ts = _globaltimer_ns()
                close_id = self._open[close_no]
                if close_id < self.max_events:
                    self.prof.buffer[self._rec(close_id) + EV_EN] = boundary_ts
                open_id = self._alloc(open_no, boundary_ts)
            self._open[open_no] = open_id
            return open_id
        return None


@dataclass(frozen=True)
class ProfilerArgs:
    """Device-side profiler handle bundling the buffer and the enable flag.

    Passed as a SINGLE kernel argument. The jit-arg protocol below sends only
    ``buffer`` across the kernel boundary as an MLIR value; ``enabled`` stays a
    compile-time constant so ``const_expr(prof.enabled)`` folds every profiler
    call to nothing when off, and a ``None`` buffer adds no device argument.
    Open a session inside the kernel with ``prof.begin()``.
    """

    buffer: cute.Tensor
    enabled: cutlass.Constexpr[bool]
    # When on, each event records a second timestamp (``EV_ST_LATE`` /
    # ``EV_EN_EARLY``) for the overhead bands. Compile-time constant, folds away
    # when off.
    show_overhead: cutlass.Constexpr[bool] = False

    def __extract_mlir_values__(self):
        return [] if self.buffer is None else self.buffer.__extract_mlir_values__()

    def __new_from_mlir_values__(self, values):
        if self.buffer is None:
            return ProfilerArgs(self.buffer, self.enabled, self.show_overhead)
        return ProfilerArgs(self.buffer.__new_from_mlir_values__(values), self.enabled, self.show_overhead)

    @cute.jit
    def begin(self) -> Session:
        """Open a profiling session: read the header + this warp's ids ONCE.

        Also records this warp's ``sm_id`` once (constant for the warp's
        lifetime), so per-event records need not carry it. No-op when disabled.
        """
        if cutlass.const_expr(self.enabled):
            num_blocks = self.buffer[HDR_NUM_BLOCKS]
            num_warps = self.buffer[HDR_NUM_WARPS]
            max_events = self.buffer[HDR_MAX_EVENTS]
            # Row-major linear ids, general for 1-/2-/3-D grids and blocks: the
            # formula uses only the X/Y extents plus the Z coordinate (unused dims
            # are 1 and fall out).
            bidx, bidy, bidz = cute.arch.block_idx()
            gdx, gdy, _ = cute.arch.grid_dim()
            block_linear = (bidz * gdy + bidy) * gdx + bidx
            tidx, tidy, tidz = cute.arch.thread_idx()
            bdx, bdy, _ = cute.arch.block_dim()
            warp = ((tidz * bdy + tidy) * bdx + tidx) // 32
            warp_global = block_linear * num_warps + warp
            total_warps = num_blocks * num_warps
            # Layout: header | smid[tw] | counters[tw*STRIDE] | events[tw*E*F].
            # Counters are line-padded (COUNTER_STRIDE per warp) to avoid cross-warp
            # false sharing; only slot 0 of each stride is written.
            smid_off = HEADER + warp_global
            counters_base = HEADER + total_warps
            events_base = HEADER + total_warps + total_warps * COUNTER_STRIDE
            if cute.arch.lane_idx() == 0:  # one sm_id write per warp, not per event
                self.buffer[smid_off] = _smid()
            return Session(self, warp_global, max_events, counters_base, events_base)
        return Session(self, None, None, None, None)


# ---------------------------------------------------------------------------
# Host-side management, parsing and export
# ---------------------------------------------------------------------------
def make_cute_buffer(buffer: torch.Tensor) -> cute.Tensor:
    """Wrap the 1-D int64 profiler buffer as a (static-layout) CuTe tensor."""
    return from_dlpack(buffer, assumed_align=16)


class SmProfiler:
    """Host-side owner of the profiler buffer and its readout.

    Mirrors the original ``sm_profiler_create_buffer`` / ``register_event`` /
    ``export_to_file`` API.

    Parameters
    ----------
    num_blocks:
        Number of CTAs in the launch grid.
    num_warps:
        Warps per block (the profiler's per-block "warp" count).
    max_events_per_warp:
        Per-warp event capacity; events beyond it are dropped.
    """

    PRINT_WIDTH = 100

    def __init__(self, num_blocks: int, num_warps: int, max_events_per_warp: int):
        self.num_blocks = int(num_blocks)
        self.num_warps = int(num_warps)
        self.max_events = int(max_events_per_warp)
        self.total_warps = self.num_blocks * self.num_warps
        self.names: dict[int, str] = {}
        # header + smid[total_warps] + counters[total_warps*COUNTER_STRIDE] +
        # events[total_warps*max_events*F]. Counters are line-padded to avoid
        # cross-warp false sharing on the per-event RMW.
        size = HEADER + self.total_warps * (1 + COUNTER_STRIDE + self.max_events * NUM_EV_FIELDS)
        self.buffer = torch.zeros(size, dtype=torch.int64, device="cuda")
        self._write_header()

    def _write_header(self) -> None:
        self.buffer[HDR_NUM_BLOCKS] = self.num_blocks
        self.buffer[HDR_NUM_WARPS] = self.num_warps
        self.buffer[HDR_MAX_EVENTS] = self.max_events

    def reset(self) -> None:
        self.buffer.zero_()
        self._write_header()

    def register_event(self, event_no: int, name: str) -> None:
        """Name an event type for the readout (host-side registry)."""
        self.names[int(event_no)] = str(name)

    def cute_buffer(self) -> cute.Tensor:
        """CuTe view of the buffer, for compile templates / kernel launch."""
        return make_cute_buffer(self.buffer)

    # ----- parsing -----
    def _records(self):
        """Yield a :class:`Record` per written event (skips unclosed/malformed)."""
        buf = self.buffer.cpu()
        # Layout: header | smid[total_warps] | counters[total_warps*COUNTER_STRIDE] | events[..].
        smid = buf[HEADER : HEADER + self.total_warps].reshape(self.num_blocks, self.num_warps)
        counters_end = HEADER + self.total_warps + self.total_warps * COUNTER_STRIDE
        # Each warp's counter occupies COUNTER_STRIDE slots; only slot 0 is used.
        counters = buf[HEADER + self.total_warps : counters_end].reshape(
            self.num_blocks, self.num_warps, COUNTER_STRIDE
        )[:, :, 0]
        ev_base = counters_end
        events = buf[ev_base:].reshape(self.num_blocks, self.num_warps, self.max_events, NUM_EV_FIELDS)
        for block in range(self.num_blocks):
            for warp in range(self.num_warps):
                sm_id = int(smid[block, warp].item())  # per-warp, recorded once
                n = min(int(counters[block, warp].item()), self.max_events)
                for event_id in range(n):
                    rec = events[block, warp, event_id]
                    st, en = int(rec[EV_ST].item()), int(rec[EV_EN].item())
                    if en <= 0 or en < st:  # unclosed / malformed
                        continue
                    yield Record(
                        block=block,
                        warp=warp,
                        event_id=event_id,
                        event_no=int(rec[EV_EVENT_NO].item()),
                        sm_id=sm_id,
                        st=st,
                        en=en,
                        st_late=int(rec[EV_ST_LATE].item()),
                        en_early=int(rec[EV_EN_EARLY].item()),
                    )

    # ----- summary (generic: per event type) -----
    def summarize(self, tag: str) -> None:
        """Per-event-type duration stats (ns, globaltimer) over all (block, warp).

        ``en - st`` is the interval (work + instrumentation), since the stamps
        are stretched to cover bookkeeping. With ``show_overhead`` recorded, the
        inner ``[EV_ST_LATE, EV_EN_EARLY]`` span (work-only) is added as a column.
        """
        intervals: dict[int, list] = {}
        works: dict[int, list] = {}
        for record in self._records():
            intervals.setdefault(record.event_no, []).append(record.en - record.st)
            if record.st_late > 0 and record.en_early > 0 and record.en_early >= record.st_late:
                works.setdefault(record.event_no, []).append(record.en_early - record.st_late)
        have_work = bool(works)
        print(
            f" Profile ({tag}): {self.num_blocks} blocks x {self.num_warps} warps, units = ns (globaltimer) ".center(
                self.PRINT_WIDTH, "-"
            )
        )
        header = f"{'event':<14}{'min':>13}{'mean':>13}{'max':>13}{'samples':>10}"
        if have_work:
            header += f"{'work_mean':>13}"
        print(header)
        for event_no in sorted(intervals):
            vals = intervals[event_no]
            name = self.names.get(event_no, f"event_{event_no}")
            t = torch.tensor(vals, dtype=torch.float64)
            line = (
                f"{name:<14}{int(t.min().item()):>13d}{int(t.mean().item()):>13d}"
                f"{int(t.max().item()):>13d}{len(vals):>10d}"
            )
            if have_work:
                work_vals = works.get(event_no)
                if work_vals:
                    work_mean = int(torch.tensor(work_vals, dtype=torch.float64).mean().item())
                    line += f"{work_mean:>13d}"
                else:
                    line += f"{'--':>13}"
            print(line)

    # ----- export -----
    def export_perfetto(self, path: str, show_overhead: bool = False) -> int:
        """Write a Chrome-Trace JSON (open in https://ui.perfetto.dev).

        Each range event becomes one ``ph:"X"`` slice over the full ``[EV_ST,
        EV_EN]`` interval; ``ts`` / ``dur`` are microseconds rebased to the
        earliest start.

        One Chrome "process" per block, one "thread" per warp. ``tid`` is kept
        disjoint from every ``pid`` because Perfetto's importer folds a thread
        whose ``tid`` equals its ``pid`` into the process's "main thread": hence
        ``pid = block + 1`` and ``tid = tid_base + block*num_warps + warp`` with
        ``tid_base`` above the pid range.

        Under ``show_overhead``, each event also emits ``Δstart`` /``Δend`` child
        slices over the two bands (``overhead`` category), each only when its
        inner stamp is present and the band has positive width.
        """
        records = list(self._records())
        base = min((record.st for record in records), default=0)
        tid_base = self.num_blocks + 1

        def pid_of(block):
            return block + 1

        def tid_of(block, warp):
            return tid_base + block * self.num_warps + warp

        def us(ts_ns):
            return (ts_ns - base) / 1000.0

        sm_of = {}
        for record in records:
            sm_of.setdefault((record.block, record.warp), record.sm_id)  # first event's SM

        meta = []
        for block in range(self.num_blocks):
            pid = pid_of(block)
            sm_id = sm_of.get((block, 0), -1)
            meta.append(
                {"name": "process_name", "ph": "M", "pid": pid, "args": {"name": f"block {block} (sm {sm_id})"}}
            )
            meta.append({"name": "process_sort_index", "ph": "M", "pid": pid, "args": {"sort_index": block}})
            for warp in range(self.num_warps):
                tid = tid_of(block, warp)
                meta.append(
                    {"name": "thread_name", "ph": "M", "pid": pid, "tid": tid, "args": {"name": f"warp {warp}"}}
                )
                meta.append(
                    {"name": "thread_sort_index", "ph": "M", "pid": pid, "tid": tid, "args": {"sort_index": warp}}
                )

        slices = []
        for record in records:
            name = self.names.get(record.event_no, f"event_{record.event_no}")
            pid, tid = pid_of(record.block), tid_of(record.block, record.warp)
            slices.append(
                {
                    "name": name,
                    "cat": name,
                    "ph": "X",
                    "ts": us(record.st),
                    "dur": (record.en - record.st) / 1000.0,
                    "pid": pid,
                    "tid": tid,
                    "args": {"sm_id": record.sm_id},
                }
            )
            if not show_overhead:
                continue
            # Start-side band [EV_ST, EV_ST_LATE]: profiler bookkeeping cost.
            if record.st_late > record.st:
                slices.append(
                    {
                        "name": f"{name} Δstart",
                        "cat": "overhead",
                        "ph": "X",
                        "ts": us(record.st),
                        "dur": (record.st_late - record.st) / 1000.0,
                        "pid": pid,
                        "tid": tid,
                        "args": {"sm_id": record.sm_id},
                    }
                )
            # End-side band [EV_EN_EARLY, EV_EN]: trailing store cost.
            if 0 < record.en_early < record.en:
                slices.append(
                    {
                        "name": f"{name} Δend",
                        "cat": "overhead",
                        "ph": "X",
                        "ts": us(record.en_early),
                        "dur": (record.en - record.en_early) / 1000.0,
                        "pid": pid,
                        "tid": tid,
                        "args": {"sm_id": record.sm_id},
                    }
                )
        trace = {
            "traceEvents": meta + slices,
            "displayTimeUnit": "ns",
            "metadata": {"unit": "globaltimer ns -> us, grid-relative"},
        }
        with open(path, "w") as handle:
            json.dump(trace, handle)
        return len(slices)
