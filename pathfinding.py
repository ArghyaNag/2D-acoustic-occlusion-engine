from __future__ import annotations
import heapq
import math
from dataclasses import dataclass, field
from typing import Optional
from shared_state import GRID_COLS, GRID_ROWS
_SQRT2: float = math.sqrt(2)
_DIRECTIONS: list[tuple[int, int, float]] = [(0, -1, 1.0), (0, 1, 1.0), (-1, 0, 1.0), (1, 0, 1.0), (-1, -1, _SQRT2), (1, -1, _SQRT2), (-1, 1, _SQRT2), (1, 1, _SQRT2)]

@dataclass
class PathResult:
    cells: list[tuple[int, int]]
    length: float
    corners: int

def _heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return max(dx, dy) + (_SQRT2 - 1.0) * min(dx, dy)

def _a_star(start: tuple[int, int], end: tuple[int, int], walls: set[tuple[int, int]] | frozenset[tuple[int, int]], grid_cols: int, grid_rows: int, blocked_nodes: frozenset[tuple[int, int]]=frozenset(), blocked_root_edges: frozenset[tuple[tuple[int, int], tuple[int, int]]]=frozenset()) -> Optional[list[tuple[int, int]]]:
    endpoints = {start, end}

    def is_blocked(cell: tuple[int, int]) -> bool:
        if cell in endpoints:
            return False
        return cell in walls or cell in blocked_nodes
    counter = 0
    open_set: list[tuple[float, int, tuple[int, int]]] = []
    heapq.heappush(open_set, (_heuristic(start, end), counter, start))
    counter += 1
    g_score: dict[tuple[int, int], float] = {start: 0.0}
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    while open_set:
        f, _, current = heapq.heappop(open_set)
        if current == end:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path
        current_g = g_score.get(current, math.inf)
        if f - _heuristic(current, end) > current_g + 1e-09:
            continue
        cx, cy = current
        for dx, dy, step_cost in _DIRECTIONS:
            nx, ny = (cx + dx, cy + dy)
            neighbour = (nx, ny)
            if nx < 0 or nx >= grid_cols or ny < 0 or (ny >= grid_rows):
                continue
            if is_blocked(neighbour):
                continue
            if (current, neighbour) in blocked_root_edges:
                continue
            tentative_g = current_g + step_cost
            if tentative_g < g_score.get(neighbour, math.inf):
                g_score[neighbour] = tentative_g
                came_from[neighbour] = current
                f_new = tentative_g + _heuristic(neighbour, end)
                heapq.heappush(open_set, (f_new, counter, neighbour))
                counter += 1
    return None

def _path_length(path: list[tuple[int, int]]) -> float:
    total = 0.0
    for i in range(1, len(path)):
        dx = path[i][0] - path[i - 1][0]
        dy = path[i][1] - path[i - 1][1]
        total += math.sqrt(dx * dx + dy * dy)
    return total

def _count_corners(path: list[tuple[int, int]]) -> int:
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
    return PathResult(cells=path, length=_path_length(path), corners=_count_corners(path))

def find_k_paths(start: tuple[int, int], end: tuple[int, int], walls: set[tuple[int, int]] | frozenset[tuple[int, int]], grid_cols: int, grid_rows: int, k: int=1) -> list[PathResult]:
    start = (int(round(start[0])), int(round(start[1])))
    end = (int(round(end[0])), int(round(end[1])))
    if start == end:
        return [PathResult(cells=[start], length=0.0, corners=0)]
    shortest = _a_star(start, end, walls, grid_cols, grid_rows)
    if shortest is None:
        return []
    a_paths: list[list[tuple[int, int]]] = [shortest]
    b_candidates: list[tuple[float, int, list[tuple[int, int]]]] = []
    b_counter = 0
    seen: set[tuple[tuple[int, int], ...]] = {tuple(shortest)}
    for i in range(1, k):
        prev_path = a_paths[i - 1]
        for j in range(len(prev_path) - 1):
            spur_node = prev_path[j]
            root_path = prev_path[:j + 1]
            blocked_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
            for a_path in a_paths:
                if len(a_path) > j and a_path[:j + 1] == root_path:
                    blocked_edges.add((spur_node, a_path[j + 1]))
            blocked_nodes = frozenset(root_path[:j])
            spur_path = _a_star(spur_node, end, walls, grid_cols, grid_rows, blocked_nodes=blocked_nodes, blocked_root_edges=frozenset(blocked_edges))
            if spur_path is not None:
                total_path = root_path[:-1] + spur_path
                path_key = tuple(total_path)
                if path_key not in seen:
                    seen.add(path_key)
                    total_length = _path_length(total_path)
                    heapq.heappush(b_candidates, (total_length, b_counter, total_path))
                    b_counter += 1
        if not b_candidates:
            break
        _, _, best = heapq.heappop(b_candidates)
        a_paths.append(best)
    return [_make_result(p) for p in a_paths]
