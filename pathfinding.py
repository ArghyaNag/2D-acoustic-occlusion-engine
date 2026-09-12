"""
pathfinding.py — Grid-based k-shortest-path routing for the audio sandbox.

This module is COMPLETELY ISOLATED from Pygame, sounddevice, NumPy, SciPy,
and threading.  Pure Python standard library only — it solves a graph/routing
problem, not a DSP problem.

Movement model
--------------
**8-directional** (N, S, E, W, NE, NW, SE, SW).

Rationale: the existing DSP code (dsp_engine.py) uses straight-line Euclidean
distance between listener and source for gain attenuation, and Bresenham ray
casting for wall-obstruction detection.  Both of these work in continuous
2D space where diagonal relationships are natural.  If pathfinding were
restricted to 4-directional movement, the path lengths it reports would
systematically over-estimate the "true" acoustic distance (a source that is
diagonally adjacent at Euclidean distance ~1.41 would report path length 2.0
via two orthogonal steps).  8-directional movement produces path lengths that
more closely approximate the Euclidean distances the DSP code already uses,
making the eventual swap from straight-line to routed distance less jarring
perceptually.

For a course project where the grid is only 40×24 = 960 cells, the added
complexity of 8 neighbours vs 4 is negligible.

Step costs: orthogonal steps cost 1.0, diagonal steps cost sqrt(2) ≈ 1.4142.
The ``length`` field on PathResult is the sum of these step costs along the
path, so it approximates true Euclidean distance through the grid.

K-shortest-paths strategy
-------------------------
**Yen's algorithm** (1971).

Rationale: the grid is small (≤960 cells) and k will typically be 2–3.  Yen's
algorithm finds the true k shortest loopless paths, which is important for
correctness — a simpler "penalise-and-rerun" heuristic can miss valid
alternatives or return duplicates.  The implementation below uses A* (with
Euclidean heuristic) as the single-source shortest-path subroutine for
efficiency, but BFS-Dijkstra would also be fine at this scale.  Yen's is
well-documented, deterministic, and straightforward to verify — all
desirable properties for a course project.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Optional

from shared_state import GRID_COLS, GRID_ROWS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SQRT2: float = math.sqrt(2)

# 8-directional neighbours: (dx, dy, step_cost)
_DIRECTIONS: list[tuple[int, int, float]] = [
    ( 0, -1, 1.0),     # N
    ( 0,  1, 1.0),     # S
    (-1,  0, 1.0),     # W
    ( 1,  0, 1.0),     # E
    (-1, -1, _SQRT2),  # NW
    ( 1, -1, _SQRT2),  # NE
    (-1,  1, _SQRT2),  # SW
    ( 1,  1, _SQRT2),  # SE
]


# ---------------------------------------------------------------------------
# PathResult dataclass
# ---------------------------------------------------------------------------
@dataclass
class PathResult:
    """Result of a single path query.

    Attributes
    ----------
    cells : list[tuple[int, int]]
        Ordered list of grid cells from start to end, inclusive.
    length : float
        Summed Euclidean step distances along the path (orthogonal = 1.0,
        diagonal = sqrt(2)).  Suitable for direct use as a distance-gain
        input in dsp_engine.py.
    corners : int
        Number of direction changes along the path.  The first step never
        counts as a corner (there is no prior direction to differ from).
    """
    cells: list[tuple[int, int]]
    length: float
    corners: int


# ---------------------------------------------------------------------------
# Internal: A* shortest path with optional node/edge exclusions for Yen's
# ---------------------------------------------------------------------------
def _heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Chebyshev-weighted Euclidean heuristic for 8-directional A*.

    Uses the octile distance: max(|dx|, |dy|) + (sqrt(2) - 1) * min(|dx|, |dy|).
    This is admissible and consistent for 8-directional grids with
    orthogonal cost 1 and diagonal cost sqrt(2).
    """
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return max(dx, dy) + (_SQRT2 - 1.0) * min(dx, dy)


def _a_star(
    start: tuple[int, int],
    end: tuple[int, int],
    walls: set[tuple[int, int]] | frozenset[tuple[int, int]],
    grid_cols: int,
    grid_rows: int,
    blocked_nodes: frozenset[tuple[int, int]] = frozenset(),
    blocked_root_edges: frozenset[tuple[tuple[int, int], tuple[int, int]]] = frozenset(),
) -> Optional[list[tuple[int, int]]]:
    """A* shortest path on an 8-directional grid.

    Parameters
    ----------
    blocked_nodes :
        Nodes that are forbidden (used by Yen's spur-path computation).
        Does NOT include *start* or *end* — those are never blocked.
    blocked_root_edges :
        Directed edges (from, to) that are forbidden at the spur node
        (used by Yen's to avoid re-discovering already-found paths).

    Returns None if no path exists.

    Note on endpoint wall handling: if *start* or *end* is itself a wall
    cell, we still allow the path to pass through it.  This mirrors the
    "exclude endpoints from the obstruction check" convention used by
    dsp_engine.py's _line_intersects_walls — standing on a wall cell
    shouldn't self-block.
    """
    # Build the effective wall set: walls minus the endpoints (so endpoints
    # are never treated as blocked even if they are in the walls set).
    # We also add blocked_nodes from Yen's, but never block start/end.
    endpoints = {start, end}

    # Fast membership check combining walls + blocked_nodes, minus endpoints.
    def is_blocked(cell: tuple[int, int]) -> bool:
        if cell in endpoints:
            return False
        return cell in walls or cell in blocked_nodes

    # Priority queue: (f_score, counter, node)
    # Counter breaks ties deterministically.
    counter = 0
    open_set: list[tuple[float, int, tuple[int, int]]] = []
    heapq.heappush(open_set, (_heuristic(start, end), counter, start))
    counter += 1

    g_score: dict[tuple[int, int], float] = {start: 0.0}
    came_from: dict[tuple[int, int], tuple[int, int]] = {}

    while open_set:
        f, _, current = heapq.heappop(open_set)

        if current == end:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        current_g = g_score.get(current, math.inf)

        # Skip stale entries (we may have pushed a better path since)
        if f - _heuristic(current, end) > current_g + 1e-9:
            continue

        cx, cy = current
        for dx, dy, step_cost in _DIRECTIONS:
            nx, ny = cx + dx, cy + dy
            neighbour = (nx, ny)

            # Bounds check
            if nx < 0 or nx >= grid_cols or ny < 0 or ny >= grid_rows:
                continue

            # Wall / blocked-node check
            if is_blocked(neighbour):
                continue

            # Blocked-edge check (Yen's)
            if (current, neighbour) in blocked_root_edges:
                continue

            tentative_g = current_g + step_cost

            if tentative_g < g_score.get(neighbour, math.inf):
                g_score[neighbour] = tentative_g
                came_from[neighbour] = current
                f_new = tentative_g + _heuristic(neighbour, end)
                heapq.heappush(open_set, (f_new, counter, neighbour))
                counter += 1

    return None  # No path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _path_length(path: list[tuple[int, int]]) -> float:
    """Compute the summed Euclidean step distance along *path*."""
    total = 0.0
    for i in range(1, len(path)):
        dx = path[i][0] - path[i - 1][0]
        dy = path[i][1] - path[i - 1][1]
        total += math.sqrt(dx * dx + dy * dy)
    return total


def _count_corners(path: list[tuple[int, int]]) -> int:
    """Count direction changes along *path*.

    A "corner" is a step whose direction (dx, dy) differs from the
    previous step's direction.  The first step has no predecessor, so
    it never counts as a corner.
    """
    if len(path) <= 2:
        return 0

    corners = 0
    prev_dx = path[1][0] - path[0][0]
    prev_dy = path[1][1] - path[0][1]
    for i in range(2, len(path)):
        dx = path[i][0] - path[i - 1][0]
        dy = path[i][1] - path[i - 1][1]
        if dx != prev_dx or dy != prev_dy:
            corners += 1
        prev_dx = dx
        prev_dy = dy
    return corners


def _make_result(path: list[tuple[int, int]]) -> PathResult:
    """Build a PathResult from a raw cell list."""
    return PathResult(
        cells=path,
        length=_path_length(path),
        corners=_count_corners(path),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def find_k_paths(
    start: tuple[int, int],
    end: tuple[int, int],
    walls: set[tuple[int, int]] | frozenset[tuple[int, int]],
    grid_cols: int,
    grid_rows: int,
    k: int = 1,
) -> list[PathResult]:
    """Find up to *k* shortest loopless paths from *start* to *end* on the grid.

    Uses Yen's algorithm with A* as the single-source shortest-path
    subroutine.  See module docstring for movement model and design
    rationale.

    Parameters
    ----------
    start, end : tuple[int, int]
        Grid-cell coordinates (integer).  If these happen to be passed as
        floats (e.g. from dsp_engine.py's float() casts), they are rounded
        to the nearest int at the boundary.
    walls : set or frozenset of (int, int)
        Grid cells that are impassable (except at the endpoints — see
        module docstring).
    grid_cols, grid_rows : int
        Grid dimensions.  Typically GRID_COLS=40, GRID_ROWS=24.
    k : int
        Maximum number of shortest paths to return.

    Returns
    -------
    list[PathResult]
        Up to *k* paths, sorted by increasing length.  Returns an EMPTY
        list if no path exists (caller should interpret this as total
        occlusion).  Never raises an exception for well-formed inputs.
    """
    # --- Boundary: round to int if caller passed floats ---------------
    # dsp_engine.py casts grid positions to float() before calling into
    # the DSP pipeline.  We round back to int here since pathfinding
    # operates on discrete grid cells.
    start = (int(round(start[0])), int(round(start[1])))
    end = (int(round(end[0])), int(round(end[1])))

    # --- Trivial case: start == end -----------------------------------
    if start == end:
        return [PathResult(cells=[start], length=0.0, corners=0)]

    # --- Yen's algorithm ----------------------------------------------
    # Step 1: Find the single shortest path.
    shortest = _a_star(start, end, walls, grid_cols, grid_rows)
    if shortest is None:
        return []  # No path exists at all.

    # A will hold the k-shortest paths found so far (sorted by length).
    a_paths: list[list[tuple[int, int]]] = [shortest]

    # B is a min-heap of candidate paths: (length, tiebreaker, path)
    b_candidates: list[tuple[float, int, list[tuple[int, int]]]] = []
    b_counter = 0

    # Set of path tuples already in A or B, to avoid exact duplicates.
    seen: set[tuple[tuple[int, int], ...]] = {tuple(shortest)}

    for i in range(1, k):
        # The most recently added path to A.
        prev_path = a_paths[i - 1]

        for j in range(len(prev_path) - 1):
            # Spur node is the j-th node of prev_path.
            spur_node = prev_path[j]

            # Root path is prev_path[0..j].
            root_path = prev_path[:j + 1]

            # Determine edges to block: for every path already in A that
            # shares the same root_path prefix, block the edge leaving
            # the spur node along that path.
            blocked_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
            for a_path in a_paths:
                if len(a_path) > j and a_path[:j + 1] == root_path:
                    # Block the edge spur_node -> a_path[j+1]
                    blocked_edges.add((spur_node, a_path[j + 1]))

            # Block all nodes in the root path except the spur node
            # itself (and start/end, which _a_star handles).
            blocked_nodes = frozenset(root_path[:j])

            # Find spur path from spur_node to end.
            spur_path = _a_star(
                spur_node, end, walls, grid_cols, grid_rows,
                blocked_nodes=blocked_nodes,
                blocked_root_edges=frozenset(blocked_edges),
            )

            if spur_path is not None:
                # Total path = root_path + spur_path (minus the shared spur_node).
                total_path = root_path[:-1] + spur_path
                path_key = tuple(total_path)
                if path_key not in seen:
                    seen.add(path_key)
                    total_length = _path_length(total_path)
                    heapq.heappush(
                        b_candidates,
                        (total_length, b_counter, total_path),
                    )
                    b_counter += 1

        if not b_candidates:
            break  # No more candidate paths.

        # Pop the shortest candidate from B and add to A.
        _, _, best = heapq.heappop(b_candidates)
        a_paths.append(best)

    return [_make_result(p) for p in a_paths]
