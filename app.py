from flask import Flask, request, jsonify, make_response
from math import hypot
from typing import Any, Dict, List, Tuple, Optional, Set
from scipy.stats import linregress
from scipy import interpolate
import numpy as np
from flask_cors import CORS
import random

try:
    # When running as a package: `python -m ubs_server.app` or `flask run` with APP=ubs_server.app
    from .utils import roman_to_int, parse_english_number, parse_german_number, chinese_to_int, classify_representation
except Exception:
    # When running directly: `python app.py`
    from utils import roman_to_int, parse_english_number, parse_german_number, chinese_to_int, classify_representation
from collections import defaultdict, deque, Counter
import re, math
import json

app = Flask(__name__)
CORS(app)

SIZE = 4
TARGET = 2048

def validate_grid(grid):
    if not isinstance(grid, list) or len(grid) != SIZE:
        return False
    for row in grid:
        if not isinstance(row, list) or len(row) != SIZE:
            return False
        for cell in row:
            if cell is not None and (not isinstance(cell, int) or cell <= 0):
                return False
    return True

def rotate_grid(grid):
    # 90 deg clockwise
    return [[grid[SIZE-1-r][c] for r in range(SIZE)] for c in range(SIZE)]

def rotate_times(grid, times):
    times = (times % 4 + 4) % 4
    g = grid
    for _ in range(times):
        g = rotate_grid(g)
    return g

def merge_row_left(row):
    nums = [v for v in row if v is not None]
    merged = []
    score_gain = 0
    i = 0
    while i < len(nums):
        if i+1 < len(nums) and nums[i] == nums[i+1]:
            val = nums[i] * 2
            merged.append(val)
            score_gain += val
            i += 2
        else:
            merged.append(nums[i])
            i += 1
    while len(merged) < SIZE:
        merged.append(None)
    return merged, score_gain

def move_left(grid):
    out = []
    moved = False
    score_gain = 0
    for r in range(SIZE):
        before = grid[r]
        row, gain = merge_row_left(before)
        out.append(row)
        score_gain += gain
        if not moved:
            if any(row[c] != before[c] for c in range(SIZE)):
                moved = True
    return out, moved, score_gain

def apply_move(grid, direction):
    if direction == 'LEFT':
        times_in, times_out = 0, 0
    elif direction == 'UP':
        times_in, times_out = 3, 1
    elif direction == 'RIGHT':
        times_in, times_out = 2, 2
    elif direction == 'DOWN':
        times_in, times_out = 1, 3
    else:
        raise ValueError('Invalid direction')
    rotated = rotate_times(grid, times_in)
    moved_grid, moved, gain = move_left(rotated)
    restored = rotate_times(moved_grid, times_out)
    return restored, moved, gain

def empty_cells(grid):
    return [(r, c) for r in range(SIZE) for c in range(SIZE) if grid[r][c] is None]

def spawn_random_tile(grid):
    cells = empty_cells(grid)
    if not cells:
        return grid
    r, c = random.choice(cells)
    val = 2 if random.random() < 0.9 else 4
    g = [row[:] for row in grid]
    g[r][c] = val
    return g

def has_2048(grid):
    return any(grid[r][c] == TARGET for r in range(SIZE) for c in range(SIZE))

def any_moves_available(grid):
    if empty_cells(grid):
        return True
    for r in range(SIZE):
        for c in range(SIZE):
            v = grid[r][c]
            if r+1 < SIZE and grid[r+1][c] == v:
                return True
            if c+1 < SIZE and grid[r][c+1] == v:
                return True
    return False

@app.route('/2048', methods=['POST'])
def play():
    data = request.get_json(silent=True) or {}
    grid = data.get('grid')
    direction = data.get('mergeDirection')

    if not validate_grid(grid):
        return jsonify(error='Invalid grid'), 400
    if direction not in ('UP', 'DOWN', 'LEFT', 'RIGHT'):
        return jsonify(error='Invalid mergeDirection'), 400

    moved_grid, did_move, _ = apply_move(grid, direction)
    next_grid = spawn_random_tile(moved_grid) if did_move else moved_grid

    end_game = None
    if has_2048(next_grid):
        end_game = 'win'
    elif not any_moves_available(next_grid):
        end_game = 'lose'

    return jsonify(nextGrid=next_grid, endGame=end_game)

# Import numeral helpers from utils
# from utils import roman_to_int, parse_english_number, parse_german_number, chinese_to_int, classify_representation
app.config['JSON_SORT_KEYS'] = False
app.json.sort_keys = False

def bad_request(message: str, details: Optional[Dict[str, Any]] = None, status_code: int = 400):
    payload = {"error": message}
    if details:
        payload["details"] = details
    resp = make_response(jsonify(payload), status_code)
    resp.headers["Content-Type"] = "application/json"
    return resp


@app.route("/sailing-club", methods=["POST"])
def sailing_club_submission():
    data = request.get_json(silent=True) or {}
    test_cases = data.get("testCases", [])
    if not isinstance(test_cases, list):
        return bad_request("Expected 'testCases' to be a list.")

    def parse_slots(slots_raw):
        valid = []
        if not isinstance(slots_raw, list):
            return valid
        for pair in slots_raw:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                try:
                    s = int(pair[0])
                    e = int(pair[1])
                    # Only accept valid intervals with positive duration
                    # Also check bounds: 0 <= hours <= 4096
                    if s < e and 0 <= s <= 4096 and 0 <= e <= 4096:
                        valid.append([s, e])
                except (ValueError, TypeError, OverflowError):
                    # Skip invalid numeric values
                    continue
        return valid

    def merge_slots(slots):
        if not slots:
            return []
        slots.sort(key=lambda x: x[0])
        merged = [slots[0][:]]
        for s, e in slots[1:]:
            ls, le = merged[-1]
            if s <= le:
                if e > le:
                    merged[-1][1] = e
            else:
                merged.append([s, e])
        return merged

    def min_boats(slots):
        if not slots:
            return 0
        events = []
        for s, e in slots:
            events.append((s, 1))
            events.append((e, -1))
        # End (-1) before start (+1) at the same time
        events.sort(key=lambda t: (t[0], t[1]))
        cur = 0
        ans = 0
        for _, d in events:
            cur += d
            if cur > ans:
                ans = cur
        return ans

    solutions = []
    for case in test_cases:
        if not isinstance(case, dict):
            continue
        cid = case.get("id")
        # Always process every test case, even if id is missing/invalid
        # Convert id to string, handling None, 0, empty string, etc.
        case_id = str(cid) if cid is not None else ""
        
        raw_slots = case.get("input", [])
        slots = parse_slots(raw_slots)
        merged = merge_slots(slots)
        boats = min_boats(slots)
        solutions.append({
            "id": case_id,
            "sortedMergedSlots": merged,
            "minBoatsNeeded": boats,
        })

    resp = make_response(jsonify({"solutions": solutions}), 200)
    resp.headers["Content-Type"] = "application/json"
    return resp

@app.errorhandler(404)
def handle_404(e):
    resp = make_response(jsonify({"error": "Not Found"}), 404)
    resp.headers["Content-Type"] = "application/json"
    return resp

@app.errorhandler(405)
def handle_405(e):
    resp = make_response(jsonify({"error": "Method Not Allowed"}), 405)
    resp.headers["Content-Type"] = "application/json"
    return resp

@app.errorhandler(500)
def handle_500(e):
    resp = make_response(jsonify({"error": "Internal Server Error"}), 500)
    resp.headers["Content-Type"] = "application/json"
    return resp

def parse_float(value: Any) -> Optional[float]:
    try:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            return float(value.strip())
    except Exception:
        return None
    return None


def parse_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "1", "yes", "y"): return True
        if v in ("false", "0", "no", "n"): return False
    return None


def as_xy(pair: Any) -> Optional[Tuple[float, float]]:
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        return None
    x = parse_float(pair[0])
    y = parse_float(pair[1])
    if x is None or y is None:
        return None
    return x, y


def linear_interpolation_np(series: np.ndarray) -> np.ndarray:
    mask = np.isnan(series)
    if np.sum(~mask) == 0:
        return np.zeros_like(series)
    if np.sum(~mask) == 1:
        return np.full_like(series, series[~mask][0])
    valid_indices = np.where(~mask)[0]
    valid_values = series[~mask]

    if len(valid_values) > 2:
        slope, _, r_value, _, _ = linregress(valid_indices, valid_values)
        trend_strength = abs(r_value)
    else:
        trend_strength = 0
    
    if trend_strength > 0.7:
        return np.interp(np.arange(len(series)), valid_indices, valid_values)
    else:
        return cubic_spline(series, mask)

def cubic_spline(series: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid_indices = np.where(~mask)[0]
    valid_values = series[~mask]
    
    try:
        spline = interpolate.CubicSpline(valid_indices, valid_values, bc_type='natural')
        result = spline(np.arange(len(series)))
        return np.clip(result, -1e6, 1e6)
    except:
        return np.interp(np.arange(len(series)), valid_indices, valid_values)


def impute_series(series: List[Any]) -> List[float]:
    # Usingsing interpolation, cubic spline, Savitzky-Golay smoothing, and exponential smoothing.
    np_series = np.array([float(x) if x is not None else np.nan for x in series], dtype=float)
    n = len(np_series)
    mask = np.isnan(np_series)

    # Edge cases
    if n == 0:
        return []
    if np.sum(~mask) == 0:
        return [0.0] * n
    if np.sum(~mask) == 1:
        return [float(np_series[~mask][0])] * n

    valid_idx = np.where(~mask)[0]
    valid_vals = np_series[~mask]

    # linear interpolation
    grid = np.arange(n)
    base = np.interp(grid, valid_idx, valid_vals)

    candidates = [base]

    # Cubic spline when enough points
    if len(valid_vals) >= 4:
        try:
            spline = interpolate.CubicSpline(valid_idx, valid_vals, bc_type='natural')
            spl = spline(grid)
            candidates.append(np.clip(spl, np.min(valid_vals) - 10 * np.std(valid_vals),
                                      np.max(valid_vals) + 10 * np.std(valid_vals)))
        except Exception:
            pass

    # Savitzky-Golay smoothing
    try:
        from scipy.signal import savgol_filter
        max_win = min(101, n if n % 2 == 1 else n - 1)
        win = max(5, max_win if max_win % 2 == 1 else max_win - 1)
        poly = 3 if win > 3 else 2
        sg = savgol_filter(base, window_length=win, polyorder=min(poly, win - 1))
        candidates.append(sg)
    except Exception:
        pass

    # Simple exponential smoothing of the base
    def exp_smooth(x: np.ndarray, alpha: float = 0.25) -> np.ndarray:
        y = x.copy()
        for i in range(1, len(y)):
            y[i] = alpha * y[i] + (1 - alpha) * y[i - 1]
        return y

    candidates.append(exp_smooth(base, alpha=0.25))

    # Evaluation (lower MSE is better)
    eps = 1e-12
    mses = []
    for cand in candidates:
        mses.append(np.mean((cand[valid_idx] - valid_vals) ** 2) + eps)

    weights = np.array([1.0 / m for m in mses])
    weights = weights / np.sum(weights)

    final = np.zeros(n, dtype=float)
    for w, cand in zip(weights, candidates):
        final += w * cand

    final[valid_idx] = valid_vals

    vmin = np.min(valid_vals) - 5 * np.std(valid_vals)
    vmax = np.max(valid_vals) + 5 * np.std(valid_vals)
    final = np.clip(final, vmin, vmax)
    final = np.nan_to_num(final, nan=0.0, posinf=vmax, neginf=vmin)

    return final.tolist()


@app.route("/blankety", methods=["POST"])
def blankety():
    try:
        data = request.get_json(silent=True)
        series = data.get("series")
        imputed_series = []
        for i, s in enumerate(series):
            try:
                imputed = impute_series(s)
                imputed_series.append(imputed)
            except Exception as e:
                # Fallback to simple linear interpolation if ensemble fails
                np_series = np.array([float(x) if x is not None else np.nan for x in s])
                mask = np.isnan(np_series)
                if np.sum(~mask) >= 2:
                    valid_indices = np.where(~mask)[0]
                    valid_values = np_series[~mask]
                    imputed = np.interp(np.arange(len(np_series)), valid_indices, valid_values)
                    imputed_series.append(imputed.tolist())
                else:
                    # All values missing, use zeros
                    imputed_series.append([0.0] * 1000)
        
        resp = make_response(jsonify({"answer": imputed_series}), 200)
        resp.headers["Content-Type"] = "application/json"
        return resp
        
    except Exception as e:
        return bad_request(f"Imputation failed: {str(e)}", status_code=500)


@app.route("/trivia", methods=["GET"])
def trivia():
    res = {
        "answers": [
            3,  # zhb "Trivia!": How many challenges are there this year, which title ends with an exclamation mark?
            1,  # zhb "Ticketing Agent": What type of tickets is the ticketing agent handling?
            2,  # zhb "Blankety Blanks": How many lists and elements per list are included in the dataset you must impute?
            2,  # zhb "Princess Diaries": What's Princess Mia's cat name in the movie Princess Diaries?
            3,  # "MST Calculation": What is the average number of nodes in a test case?
            4,  # zhb "Universal Bureau of Surveillance": Which singer did not have a James Bond theme song?
            3,  # "Operation Safeguard": What is the smallest font size in our question?
            4,  # "Capture The Flag": Which of these are anagrams of the challenge name?
            4,  # zhb "Filler 1": Where has UBS Global Coding Challenge been held before?
            3   # zhb
        ]
    }
    return jsonify(res), 200

@app.route("/ticketing-agent", methods=["POST"])
def ticketing_agent():
    data = request.get_json(silent=True)
    if data is None:
        return bad_request("Invalid JSON body.")


    customers: List[Dict[str, Any]] = data.get("customers", [])
    concerts: List[Dict[str, Any]] = data.get("concerts", [])
    priority_map: Dict[str, str] = data.get("priority", {})

    concert_by_name: Dict[str, Dict[str, Any]] = {}
    concert_locations: Dict[str, Tuple[float, float]] = {}

    for c in concerts:
        name = c.get("name")
        loc = as_xy(c.get("booking_center_location"))
        if not name or loc is None:
            continue
        concert_by_name[name] = c
        concert_locations[name] = loc

    # mapping { customer_name: concert_name }
    result_map: Dict[str, str] = {}

    for cust in customers:
        cname = cust.get("name") or ""
        vip = parse_bool(cust.get("vip_status"))
        cloc = as_xy(cust.get("location"))
        card = cust.get("credit_card")

        vip_points = 100 if vip is True else 0

        card_priority_concert: Optional[str] = None
        if isinstance(card, str):
            target = priority_map.get(card)
            if isinstance(target, str) and target in concert_by_name:
                card_priority_concert = target

        distances: Dict[str, float] = {}
        if cloc is not None and concert_locations:
            cx, cy = cloc
            for name, (x, y) in concert_locations.items():
                distances[name] = hypot(cx - x, cy - y)

        latency_points: Dict[str, float] = {}
        if distances:
            d_values = list(distances.values())
            dmin, dmax = min(d_values), max(d_values)
            if dmax == dmin:
                for name in concert_by_name.keys():
                    latency_points[name] = 30.0
            else:
                span = dmax - dmin
                for name, d in distances.items():
                    latency_points[name] = 30.0 * (dmax - d) / span
        else:
            for name in concert_by_name.keys():
                latency_points[name] = 0.0

        best_name: Optional[str] = None
        best_score: float = float("-inf")
        for name in concert_by_name.keys():
            score = vip_points + latency_points.get(name, 0.0)
            if card_priority_concert == name:
                score += 50.0

            if score > best_score:
                best_score = score
                best_name = name
            elif score == best_score and best_name is not None:
                d_curr = distances.get(name, float("inf"))
                d_best = distances.get(best_name, float("inf"))
                if d_curr < d_best:
                    best_name = name
                elif d_curr == d_best and name < best_name:
                    best_name = name

        if best_name is not None:
            result_map[cname] = best_name
            
    resp = make_response(jsonify(result_map), 200)
    resp.headers["Content-Type"] = "application/json"
    return resp

def _edge_key(spy_one: str, spy_two: str) -> Tuple[str, str]:
    if spy_one <= spy_two:
        return spy_one, spy_two
    return spy_two, spy_one


def _find_bridges(nodes: Set[str], adjacency: Dict[str, Set[str]]) -> Set[frozenset]:
    discovery_time: Dict[str, int] = {}
    low_time: Dict[str, int] = {}
    parent: Dict[str, str | None] = {}
    visited: Set[str] = set()
    bridges: Set[frozenset] = set()
    time_counter: int = 0

    def depth_first_search(source: str) -> None:
        nonlocal time_counter
        visited.add(source)
        time_counter += 1
        discovery_time[source] = time_counter
        low_time[source] = time_counter

        for neighbor in adjacency.get(source, set()):
            if neighbor not in visited:
                parent[neighbor] = source
                depth_first_search(neighbor)
                low_time[source] = min(low_time[source], low_time[neighbor])

                if low_time[neighbor] > discovery_time[source]:
                    bridges.add(frozenset({source, neighbor}))
            elif parent.get(source) != neighbor:
                low_time[source] = min(low_time[source], discovery_time[neighbor])

    for node in nodes:
        if node not in visited:
            parent[node] = None
            depth_first_search(node)

    return bridges


def _extra_channels(edges: List[Dict[str, str]]) -> List[Dict[str, str]]:
    multiplicity: Counter[Tuple[str, str]] = Counter(
        _edge_key(edge["spy1"], edge["spy2"]) for edge in edges
    )

    adjacency: Dict[str, Set[str]] = defaultdict(set)
    nodes: Set[str] = set()
    for edge in edges:
        spy_one = edge["spy1"]
        spy_two = edge["spy2"]
        nodes.add(spy_one)
        nodes.add(spy_two)
        key = _edge_key(spy_one, spy_two)
        adjacency[key[0]].add(key[1])
        adjacency[key[1]].add(key[0])

    bridges: Set[frozenset] = _find_bridges(nodes, adjacency)

    non_bridge_keys: Set[frozenset] = set()
    for key, count in multiplicity.items():
        undirected = frozenset(key)
        if count > 1:
            non_bridge_keys.add(undirected)

    for node, neighbors in adjacency.items():
        for neighbor in neighbors:
            undirected = frozenset({node, neighbor})
            if undirected not in bridges:
                non_bridge_keys.add(undirected)

    extra: List[Dict[str, str]] = []
    for edge in edges:
        undirected = frozenset(_edge_key(edge["spy1"], edge["spy2"]))
        if undirected in non_bridge_keys:
            extra.append({"spy1": edge["spy1"], "spy2": edge["spy2"]})

    return extra

@app.post("/investigate")
def investigate():
    payload = request.get_json(force=True, silent=True) or {}
    if payload is None:
        try:
            raw = (request.data or b"").decode("utf-8")
            payload = json.loads(raw) if raw else {}
        except Exception:
            payload = {}
    networks_input: List[Dict] = payload.get("networks", [])

    if isinstance(payload, dict) and "body" in payload and isinstance(payload["body"], str):
        try:
            payload = json.loads(payload["body"])
        except Exception:
            payload = {}

    networks_input = payload.get("networks", []) if isinstance(payload, dict) else []

    result_networks = []
    for entry in networks_input if isinstance(networks_input, list) else []:
        # Accept multiple id field names; always return as networkId
        network_id = (
            entry.get("networkId")
            or entry.get("id")
            or entry.get("network_id")
            or entry.get("uuid")
        )

        edges = entry.get("network", [])
        extra = _extra_channels(edges if isinstance(edges, list) else [])

        result_networks.append({
            "networkId": network_id,
            "extraChannels": extra,
        })
    resp = make_response(jsonify({"networks": result_networks}), 200)
    resp.headers["Content-Type"] = "application/json"
    return resp


@app.route("/")
def root():
    return "OK", 200



@app.route("/princess-diaries", methods=["POST"])
def princess_diaries():
    data = request.get_json(silent=True)
    if not data:
        return bad_request("Invalid JSON body.")
    
    tasks = data.get("tasks", [])
    subway = data.get("subway", [])
    starting_station = data.get("starting_station")

    if not isinstance(tasks, list) or not isinstance(subway, list) or starting_station is None:
        return bad_request("Invalid input format.")
    if not tasks:
        return jsonify({"max_score": 0, "min_fee": 0, "schedule": []}), 200

    # 1) Compress stations and build adjacency with min edge weights
    stations = {starting_station}
    for t in tasks:
        stations.add(t["station"])
    all_nodes = set(stations)
    for r in subway:
        conn = r.get("connection", [])
        if len(conn) == 2:
            all_nodes.add(conn[0]); all_nodes.add(conn[1])

    idx_of = {s: i for i, s in enumerate(all_nodes)}
    N = len(all_nodes)

    adj = [dict() for _ in range(N)]
    for r in subway:
        conn = r.get("connection", [])
        fee = r.get("fee", 0)
        if len(conn) == 2:
            u, v = idx_of[conn[0]], idx_of[conn[1]]
            w = adj[u].get(v, float('inf'))
            if fee < w:
                adj[u][v] = fee
                adj[v][u] = fee
    adj = [ [(v, w) for v, w in row.items()] for row in adj ]

    # 2) Compute shortest paths only from needed sources using Dijkstra
    import heapq

    def dijkstra(src: int):
        dist = [float('inf')] * N
        dist[src] = 0
        h = [(0, src)]
        while h:
            d, u = heapq.heappop(h)
            if d != dist[u]:
                continue
            for v, w in adj[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(h, (nd, v))
        return dist

    needed_sources = {idx_of[starting_station]}
    for t in tasks:
        needed_sources.add(idx_of[t["station"]])

    dist_from = {}
    for src in needed_sources:
        dist_from[src] = dijkstra(src)

    s_idx = idx_of[starting_station]

    # 3) Sort tasks and prepare arrays
    tasks.sort(key=lambda t: t["start"])
    n = len(tasks)
    task_station = [idx_of[t["station"]] for t in tasks]
    starts = [t["start"] for t in tasks]
    ends = [t["end"] for t in tasks]
    scores = [t["score"] for t in tasks]
    names = [t["name"] for t in tasks]

    # first_compatible[i]: first j > i with starts[j] >= ends[i] (binary search)
    first_compatible = [n] * n
    for i in range(n):
        lo, hi = i + 1, n
        ei = ends[i]
        while lo < hi:
            mid = (lo + hi) // 2
            if starts[mid] >= ei:
                hi = mid
            else:
                lo = mid + 1
        first_compatible[i] = lo

    # 4) DP: best schedule starting at task i (without initial s0->i cost)
    dp_score = [0] * n
    dp_fee = [0] * n
    dp_next = [n] * n  # next index, or n for end

    for i in range(n - 1, -1, -1):
        ui = task_station[i]

        # Option A: end after i (return home)
        best_score = scores[i]
        best_fee = dist_from[ui][s_idx]
        best_next = n

        # Option B: go to any compatible next task j
        j0 = first_compatible[i]
        for j in range(j0, n):
            vj = task_station[j]
            sc = scores[i] + dp_score[j]
            fee = dist_from[ui][vj] + dp_fee[j]
            if sc > best_score or (sc == best_score and fee < best_fee):
                best_score = sc
                best_fee = fee
                best_next = j

        dp_score[i] = best_score
        dp_fee[i] = best_fee
        dp_next[i] = best_next

    # 5) Choose best starting task (add initial s0->first cost), also allow empty schedule
    max_score = 0
    min_fee = 0
    start_idx = n  # n means choose nothing

    for i in range(n):
        ui = task_station[i]
        total_score = dp_score[i]
        total_fee = dist_from[s_idx][ui] + dp_fee[i]
        if total_score > max_score or (total_score == max_score and total_fee < min_fee):
            max_score = total_score
            min_fee = total_fee
            start_idx = i

    # 6) Reconstruct schedule
    schedule = []
    i = start_idx
    while i < n:
        schedule.append(names[i])
        i = dp_next[i]

    return jsonify({"max_score": max_score, "min_fee": min_fee, "schedule": schedule}), 200

@app.route("/trading-formula", methods=["POST"]) 
def trading_formula():
    data = request.get_json(silent=True)
    if data is None:
        return bad_request("Invalid JSON body.")

    if not isinstance(data, list):
        return bad_request("Expected a list of test cases.")

    results = []

    def replace_text_commands(s: str) -> str:
        # Replace \text{VAR} with VAR
        return re.sub(r'\\text\{([^}]+)\}', lambda m: m.group(1).strip(), s)

    def replace_times_dot(s: str) -> str:
        return s.replace('\\times', '*').replace('\\cdot', '*')

    def remove_latex_wrappers(s: str) -> str:
        s = s.replace('\\left', '').replace('\\right', '')
        s = s.replace('\\,', '').replace('\\;', '').replace('\\ ', '')
        return s

    def replace_frac(s: str) -> str:
        # recursively replace simple \frac{a}{b}
        pattern = re.compile(r'\\frac\s*{([^{}]+)}\s*{([^{}]+)}')
        while True:
            m = pattern.search(s)
            if not m:
                break
            a = m.group(1)
            b = m.group(2)
            s = s[:m.start()] + f'(({a})/({b}))' + s[m.end():]
        return s

    def replace_superscripts(s: str) -> str:
        # replace {...}^{...} or ^{...}
        s = re.sub(r'\^\{([^}]+)\}', r'**(\1)', s)
        # single char superscripts like x^2
        s = re.sub(r'([A-Za-z0-9_\)\]])\^([A-Za-z0-9_\(])', r'\1**\2', s)
        return s

    def replace_exp_e(s: str) -> str:
        # e^{x} or \mathrm{e}^{x} -> exp(x)
        s = re.sub(r'e\^\{([^}]+)\}', r'exp(\1)', s)
        s = re.sub(r'\\mathrm\{e\}\^\{([^}]+)\}', r'exp(\1)', s)
        return s

    def replace_summation(s: str) -> str:
        # handle \sum_{i=START}^{END}{BODY} or without braces for BODY
        pattern = re.compile(r'\\sum_{(\w+)=(.+?)}\^\{(.+?)\}\s*\{([^}]+)\}')
        def _repl(m):
            idx = m.group(1)
            start = m.group(2)
            end = m.group(3)
            body = m.group(4)
            return f"(sum(({body}) for {idx} in range(int(({start})), int(({end}))+1)))"
        s = pattern.sub(_repl, s)
        pattern2 = re.compile(r'\\sum_{(\w+)=(.+?)}\^\{(.+?)\}\s*([^\\\s]+)')
        s = pattern2.sub(_repl, s)
        return s

    def replace_latex_commands(s: str) -> str:
        # Map common Greek letters and remove backslashes on commands
        greek = ['alpha','beta','gamma','delta','epsilon','zeta','eta','theta','iota','kappa','lambda',
                 'mu','nu','xi','omicron','pi','rho','sigma','tau','upsilon','phi','chi','psi','omega']
        for g in greek:
            s = s.replace('\\' + g, g)

        # Common function names
        s = s.replace('\\max', 'max').replace('\\min', 'min').replace('\\log', 'log').replace('\\exp', 'exp')

        # Remove other backslashes from simple commands (e.g. \beta -> beta)
        s = re.sub(r'\\([A-Za-z]+)', r'\1', s)
        return s

    def replace_subscripts_and_brackets(s: str) -> str:
        # Convert _{...} -> _... and _x -> _x (keep underscore for python identifiers)
        s = re.sub(r'_|\u2019', '_', s)  
        s = re.sub(r'_\{([^}]+)\}', lambda m: '_' + re.sub(r'\s+', '_', m.group(1)), s)
        s = re.sub(r'_([A-Za-z0-9])', r'_\1', s)

        # Convert bracketed notation A[B] -> A_B (iteratively)
        prev = None
        while prev != s:
            prev = s
            s = re.sub(r'([A-Za-z0-9_])\[([^\]]+)\]', r'\1_\2', s)
        return s

    def latex_to_python(expr: str) -> str:
        s = expr
        s = replace_text_commands(s)
        s = replace_times_dot(s)
        s = remove_latex_wrappers(s)
        s = replace_frac(s)
        s = replace_exp_e(s)
        s = replace_summation(s)
        s = replace_latex_commands(s)
        s = replace_subscripts_and_brackets(s)
        s = replace_superscripts(s)
        s = s.replace('\\log', 'log')
        # remove $ signs
        s = s.replace('$', '')
        # normalize braces to parentheses for any remaining groups
        s = s.replace('{', '(').replace('}', ')')
        # collapse multiple spaces
        s = re.sub(r'\s+', ' ', s)
        s = s.strip()

        # Insert implied multiplication where LaTeX omits the '*', e.g.:
        #   beta_i (E_R_m - R_f) -> beta_i*(E_R_m - R_f)
        #   2x -> 2*x
        #   )( -> )*(
        def insert_implied_multiplication(t: str) -> str:
            # Insert '*' at the opening bracket after a number or closing bracket
            t = re.sub(r'(?<=[0-9\)])\s*(?=\()', '*', t)
            # Insert '*' when an identifier is followed by whitespace then '('
            # Don't insert between a function name and '(' e.g. max()
            t = re.sub(r'([A-Za-z_][A-Za-z0-9_]*)\s+\(', r'\1*(', t)
            # Insert '*' between a number and identifier/underscore
            t = re.sub(r'(?<=[0-9\.])\s*(?=[A-Za-z_])', '*', t)
            # Insert '*' at the closing bracket before an identifier
            t = re.sub(r'(?<=\))\s*(?=[A-Za-z_0-9])', '*', t)
            return t

        s = insert_implied_multiplication(s)
        return s

    safe_funcs = {
        'exp': math.exp,
        'log': math.log,
        'log10': math.log10,
        'max': max,
        'min': min,
        'sum': sum,
        'pow': pow,
        'abs': abs,
        'e': math.e,
    }

    for case in data:
        try:
            formula = case.get('formula') if isinstance(case, dict) else None
            variables = case.get('variables') if isinstance(case, dict) else {}
            if formula is None or not isinstance(variables, dict):
                results.append({'result': None})
                continue

            # take RHS if formula contains =
            if '=' in formula:
                rhs = formula.split('=', 1)[1]
            else:
                rhs = formula

            py = latex_to_python(rhs)

            local_env = {}
            for k, v in variables.items():
                try:
                    local_env[str(k)] = float(v)
                except Exception:
                    local_env[str(k)] = v

            sanitized = {}
            for k in list(local_env.keys()):
                sk = k.replace('[', '_').replace(']', '').replace(' ', '_')
                if sk != k and sk not in local_env:
                    sanitized[sk] = local_env[k]
            local_env.update(sanitized)

            local_env.update(safe_funcs)

            # Evaluate expression in restricted environment
            value = eval(py, {'__builtins__': None}, local_env)

            valf = float(value)
            results.append({'result': round(valf, 4)})
        except Exception:
            results.append({'result': None})

    return jsonify(results), 200



# === FOG OF WALL ===

from typing import Tuple

# Per-(challenger_id, game_id) state
_fog_state: Dict[Tuple[str, str], Dict[str, Any]] = {}

def _in_bounds(x: int, y: int, n: int) -> bool:
    return 0 <= x < n and 0 <= y < n

def _neighbors4(x: int, y: int, n: int):
    if y - 1 >= 0: yield (x, y - 1, "N")
    if y + 1 < n: yield (x, y + 1, "S")
    if x + 1 < n: yield (x + 1, y, "E")
    if x - 1 >= 0: yield (x - 1, y, "W")

def _dir_to_vec(direction: str) -> Tuple[int, int]:
    # Grid: (0,0) is top-left; N: y-1, S: y+1, E: x+1, W: x-1
    return {
        "N": (0, -1),
        "S": (0, 1),
        "E": (1, 0),
        "W": (-1, 0),
    }.get(direction, (0, 0))

def _scan_unknown_count_at(center: Tuple[int, int], n: int, known_empty: Set[Tuple[int, int]], known_walls: Set[Tuple[int, int]]) -> int:
    cx, cy = center
    unknown = 0
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            x, y = cx + dx, cy + dy
            if not _in_bounds(x, y, n):
                continue
            if (x, y) in known_empty or (x, y) in known_walls:
                continue
            unknown += 1
    return unknown

def _update_frontier(state: Dict[str, Any], around: Optional[List[Tuple[int, int]]] = None):
    n = state["length"]
    known_empty: Set[Tuple[int, int]] = state["known_empty"]
    known_walls: Set[Tuple[int, int]] = state["known_walls"]
    frontier: Set[Tuple[int, int]] = state.setdefault("frontier", set())

    def consider_neighbors(x: int, y: int):
        for nx, ny, _ in _neighbors4(x, y, n):
            if (nx, ny) in known_empty or (nx, ny) in known_walls:
                if (nx, ny) in frontier:
                    try:
                        frontier.remove((nx, ny))
                    except KeyError:
                        pass
                continue
            frontier.add((nx, ny))

    if around:
        for cx, cy in around:
            consider_neighbors(cx, cy)
    else:
        # Rebuild from scratch
        frontier.clear()
        for (ex, ey) in list(known_empty):
            consider_neighbors(ex, ey)

def _integrate_scan(state: Dict[str, Any], crow_id: str, scan_result: List[List[str]]):
    n = state["length"]
    cx, cy = state["crows"][crow_id]

    # The scan_result is 5x5 centered at crow, rows top-to-bottom, cols left-to-right
    newly_empty: Set[Tuple[int, int]] = set()
    for r in range(5):
        for c in range(5):
            symbol = scan_result[r][c]
            x = cx + (c - 2)
            y = cy + (r - 2)
            if not _in_bounds(x, y, n):
                continue
            if symbol == "X":
                continue
            # Treat '_' as empty as well (scanner shows '_' in examples for empty)
            if symbol == "C" or symbol == "*" or symbol == "_":
                state["known_empty"].add((x, y))
                newly_empty.add((x, y))
            elif symbol == "W":
                state["known_walls"].add((x, y))
    # Always ensure the crow's current cell is empty
    state["known_empty"].add((cx, cy))
    newly_empty.add((cx, cy))
    # Mark this center as scanned and update frontier around the scanned window center
    state.setdefault("scanned_centers", set()).add((cx, cy))
    # Update frontier around all newly discovered empties
    _update_frontier(state, around=list(newly_empty))
    # Clear reservation if this center was reserved for this crow
    reservations: Dict[str, Tuple[int, int]] = state.setdefault("reservations", {})
    if reservations.get(crow_id) == (cx, cy):
        try:
            del reservations[crow_id]
        except Exception:
            pass

def _process_previous_action(state: Dict[str, Any], previous_action: Dict[str, Any]):
    if not previous_action:
        return
    action = previous_action.get("your_action")
    crow_id = previous_action.get("crow_id")
    if not crow_id or crow_id not in state["crows"]:
        return

    if action == "move":
        direction = previous_action.get("direction")
        move_result = previous_action.get("move_result") or []
        if not isinstance(move_result, list) or len(move_result) != 2:
            return
        old_x, old_y = state["crows"][crow_id]
        dx, dy = _dir_to_vec(direction or "")
        intended = (old_x + dx, old_y + dy)
        new_x, new_y = int(move_result[0]), int(move_result[1])

        # If position unchanged, we hit a wall (intended cell is a wall)
        if (new_x, new_y) == (old_x, old_y):
            ix, iy = intended
            n = state["length"]
            if _in_bounds(ix, iy, n):
                state["known_walls"].add((ix, iy))
        else:
            # Successful move: mark destination as empty
            state["known_empty"].add((new_x, new_y))

        # Update crow position to move_result (unchanged if hit wall)
        state["crows"][crow_id] = (new_x, new_y)
        _update_frontier(state, around=[(new_x, new_y)])

    elif action == "scan":
        scan_result = previous_action.get("scan_result")
        if isinstance(scan_result, list) and len(scan_result) == 5:
            _integrate_scan(state, crow_id, scan_result)

def _bfs_first_step_direction(state: Dict[str, Any], start: Tuple[int, int]) -> Tuple[Dict[Tuple[int, int], str], Dict[Tuple[int, int], int]]:
    # BFS only through known empty cells
    n = state["length"]
    known_empty = state["known_empty"]
    queue = deque()
    visited = set()
    first_dir: Dict[Tuple[int, int], str] = {}
    dist: Dict[Tuple[int, int], int] = {}

    queue.append((start[0], start[1]))
    visited.add(start)
    dist[start] = 0

    while queue:
        x, y = queue.popleft()
        for nx, ny, dir_label in _neighbors4(x, y, n):
            if (nx, ny) in visited:
                continue
            if (nx, ny) not in known_empty:
                continue
            visited.add((nx, ny))
            dist[(nx, ny)] = dist[(x, y)] + 1
            # Propagate the first step direction
            if (x, y) == start:
                first_dir[(nx, ny)] = dir_label
            else:
                first_dir[(nx, ny)] = first_dir[(x, y)]
            queue.append((nx, ny))

    return first_dir, dist

def _choose_next_action(state: Dict[str, Any]) -> Dict[str, Any]:
    # If all walls found, submit
    if len(state["known_walls"]) >= state["num_walls"]:
        submission = [f"{x}-{y}" for (x, y) in sorted(state["known_walls"])]
        return {
            "action_type": "submit",
            "submission": submission,
        }

    n = state["length"]
    total_cells = n * n
    known_cells = len(state["known_empty"]) + len(state["known_walls"])
    unknown_cells = max(0, total_cells - known_cells)
    steps = state.get("steps", 0)
    elapsed = time.time() - state.get("start_time", time.time())
    aggressive = (elapsed > 24.0) or (steps >= int(0.9 * n * n))
    reservations: Dict[str, Tuple[int, int]] = state.setdefault("reservations", {})
    reservation_set_step: Dict[str, int] = state.setdefault("reservation_set_step", {})
    # Expire stale reservations or those already scanned
    to_del = []
    for cid, cell in reservations.items():
        if cell in scanned_centers or (steps - reservation_set_step.get(cid, steps)) > 12:
            to_del.append(cid)
    for cid in to_del:
        try:
            del reservations[cid]
        except Exception:
            pass
        try:
            del reservation_set_step[cid]
        except Exception:
            pass
    scanned_centers: Set[Tuple[int, int]] = state.setdefault("scanned_centers", set())

    # Early scan seeding: perform initial scans to bootstrap knowledge
    if steps < max(2, len(state.get("crows", {}))):
        best = None
        for crow_id, (cx, cy) in state["crows"].items():
            if (cx, cy) in scanned_centers:
                continue
            gain = _scan_unknown_count_at((cx, cy), n, state["known_empty"], state["known_walls"])
            if best is None or gain > best[0]:
                best = (gain, crow_id)
        if best is not None and best[0] > 0:
            return {"action_type": "scan", "crow_id": best[1]}

    # Dynamic scan threshold based on remaining unknown area (lower in aggressive mode)
    if aggressive:
        scan_threshold = 1
    elif unknown_cells > 0.6 * total_cells:
        scan_threshold = 6
    elif unknown_cells > 0.3 * total_cells:
        scan_threshold = 3
    else:
        scan_threshold = 2

    # 1) Consider scanning now from any crow with high expected gain
    best_scan = None  # (gain, crow_id)
    # Multi-crow band bias: encourage each crow to operate in its x-band
    crow_ids_sorted = sorted(state["crows"].keys())
    num_crows = max(1, len(crow_ids_sorted))
    for crow_id, (cx, cy) in state["crows"].items():
        if (cx, cy) in scanned_centers:
            continue
        gain = _scan_unknown_count_at((cx, cy), n, state["known_empty"], state["known_walls"])
        # Apply small band bias
        try:
            band_index = crow_ids_sorted.index(crow_id)
        except ValueError:
            band_index = 0
        band_min = int(band_index * n / num_crows)
        band_max = int((band_index + 1) * n / num_crows) - 1
        if band_min <= cx <= band_max:
            gain += 1
        # Encourage scanning if this is the crow's reserved center
        if reservations.get(crow_id) == (cx, cy):
            gain += 1
        # If at reserved center and it still has any unknown gain, scan now
        if reservations.get(crow_id) == (cx, cy) and gain >= 1:
            return {
                "action_type": "scan",
                "crow_id": crow_id,
            }
        if gain >= scan_threshold:
            if best_scan is None or gain > best_scan[0]:
                best_scan = (gain, crow_id)

    if best_scan is not None:
        _, crow_id = best_scan
        return {
            "action_type": "scan",
            "crow_id": crow_id,
        }

    # 2) Move towards a reachable scan center (a known-empty cell with unknowns in 5x5)
    best_move = None  # (score_tuple, crow_id, direction)
    reserved_cells = {t for cid, t in reservations.items() if isinstance(t, tuple)}
    for crow_id, (sx, sy) in state["crows"].items():
        if (sx, sy) not in state["known_empty"]:
            # Ensure current crow cell is marked empty
            state["known_empty"].add((sx, sy))

        first_dir, dist = _bfs_first_step_direction(state, (sx, sy))
        # Evaluate all reachable known-empty cells as potential scan centers
        for cell, d in dist.items():
            if cell in scanned_centers:
                continue
            gain = _scan_unknown_count_at(cell, n, state["known_empty"], state["known_walls"])
            if gain <= 0:
                continue
            direction = first_dir.get(cell)
            if not direction:
                continue
            # Apply band bias based on candidate center x
            try:
                band_index = crow_ids_sorted.index(crow_id)
            except ValueError:
                band_index = 0
            band_min = int(band_index * n / num_crows)
            band_max = int((band_index + 1) * n / num_crows) - 1
            if band_min <= cell[0] <= band_max:
                gain += 1
            # Penalize cells reserved by other crows
            if cell in reserved_cells and reservations.get(crow_id) != cell:
                continue
            # Prefer higher gain, then shorter distance
            score = (gain, -d)
            if best_move is None or score > best_move[0]:
                best_move = (score, crow_id, direction)
                # Reserve this target center for the crow (to reduce overlap)
                reservations[crow_id] = cell
                reservation_set_step[crow_id] = steps

    if best_move is not None:
        _, crow_id, direction = best_move
        return {
            "action_type": "move",
            "crow_id": crow_id,
            "direction": direction,
        }

    # 3) Exploratory probe: step into an adjacent unknown (may hit a wall)
    best_probe = None  # (expected_gain, crow_id, direction)
    for crow_id, (cx, cy) in state["crows"].items():
        for nx, ny, direction in _neighbors4(cx, cy, n):
            if (nx, ny) in state["known_walls"]:
                continue
            if (nx, ny) in state["known_empty"]:
                continue
            # Estimate gain if we could scan from that neighbor next turn
            gain = _scan_unknown_count_at((nx, ny), n, state["known_empty"], state["known_walls"])
            # Apply band bias on neighbor center x
            try:
                band_index = crow_ids_sorted.index(crow_id)
            except ValueError:
                band_index = 0
            band_min = int(band_index * n / num_crows)
            band_max = int((band_index + 1) * n / num_crows) - 1
            if band_min <= nx <= band_max:
                gain += 1
            if best_probe is None or gain > best_probe[0]:
                best_probe = (gain, crow_id, direction)

    if best_probe is not None:
        _, crow_id, direction = best_probe
        return {
            "action_type": "move",
            "crow_id": crow_id,
            "direction": direction,
        }

    # 4) Lattice fallback: head towards an unscanned lattice center to cover edges/gaps
    # Generate coarse grid of centers
    lattice = []
    stride = 4
    for x in range(0, n, stride):
        for y in range(0, n, stride):
            lattice.append((x, y))
    # Evaluate best lattice target per crow
    best_lattice = None  # (score_tuple, crow_id, direction)
    for crow_id, (sx, sy) in state["crows"].items():
        first_dir, dist = _bfs_first_step_direction(state, (sx, sy))
        for cell in lattice:
            if cell in scanned_centers:
                continue
            if cell not in dist:
                continue
            d = dist[cell]
            gain = _scan_unknown_count_at(cell, n, state["known_empty"], state["known_walls"])
            if gain <= 0:
                continue
            direction = first_dir.get(cell)
            if not direction:
                continue
            score = (gain, -d)
            if best_lattice is None or score > best_lattice[0]:
                best_lattice = (score, crow_id, direction)

    if best_lattice is not None:
        _, crow_id, direction = best_lattice
        return {
            "action_type": "move",
            "crow_id": crow_id,
            "direction": direction,
        }

    # 5) Fallback: if absolutely nothing to do, scan with any crow (should be rare)
    any_crow = next(iter(state["crows"].keys()))
    return {
        "action_type": "scan",
        "crow_id": any_crow,
    }

@app.route("/fog-of-wall", methods=["POST"])
def fog_of_wall():
    data = request.get_json(silent=True) or {}

    challenger_id = str(data.get("challenger_id", ""))
    game_id = str(data.get("game_id", ""))
    if not challenger_id or not game_id:
        return bad_request("Missing challenger_id or game_id.")

    key = (challenger_id, game_id)

    # Initialize or update state
    test_case = data.get("test_case")
    if test_case:
        # New test case: initialize state
        length_of_grid = int(test_case.get("length_of_grid", 0))
        num_of_walls = int(test_case.get("num_of_walls", 0))
        crows_list = test_case.get("crows", [])

        if length_of_grid <= 0 or not isinstance(crows_list, list):
            return bad_request("Invalid test_case payload.")

        state = {
            "length": length_of_grid,
            "num_walls": num_of_walls,
            "known_walls": set(),         # Set[Tuple[int,int]]
            "known_empty": set(),         # Set[Tuple[int,int]]
            "crows": {},                  # Dict[str, Tuple[int,int]]
            "steps": 0,
            "start_time": time.time(),
            "scanned_centers": set(),
            "frontier": set(),
        }
        for c in crows_list:
            cid = str(c.get("id"))
            x = int(c.get("x", 0))
            y = int(c.get("y", 0))
            state["crows"][cid] = (x, y)
            if _in_bounds(x, y, length_of_grid):
                state["known_empty"].add((x, y))
        _update_frontier(state)
        _fog_state[key] = state
    else:
        # Must exist for subsequent steps
        if key not in _fog_state:
            return bad_request("State not initialized for this game_id; missing test_case.")
        state = _fog_state[key]

    # Apply previous action result to update state
    previous_action = data.get("previous_action")
    if previous_action:
        _process_previous_action(state, previous_action)
        # Count an action step after processing its result
        try:
            state["steps"] = int(state.get("steps", 0)) + 1
        except Exception:
            state["steps"] = 1

    # No early forced submit: explore aggressively via policy, submit only when complete

    # Decide next action
    next_action = _choose_next_action(state)

    # Fill required identifiers
    response = {
        "challenger_id": challenger_id,
        "game_id": game_id,
        "action_type": next_action["action_type"],
    }
    if next_action["action_type"] in ("move", "scan"):
        response["crow_id"] = next_action["crow_id"]
    if next_action["action_type"] == "move":
        response["direction"] = next_action["direction"]
    if next_action["action_type"] == "submit":
        response["submission"] = next_action["submission"]

    # Cleanup state if we are submitting
    if next_action["action_type"] == "submit":
        try:
            del _fog_state[key]
        except Exception:
            pass

    resp = make_response(jsonify(response), 200)
    resp.headers["Content-Type"] = "application/json"
    return resp

@app.route("/duolingo-sort", methods=["POST"]) 
def duolingo_sort():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return bad_request("Invalid JSON body.")

    part = data.get("part")
    challenge_input = data.get("challengeInput", {})
    if part not in ("ONE", "TWO") or not isinstance(challenge_input, dict):
        return bad_request("Invalid input format.")

    unsorted_list = challenge_input.get("unsortedList")
    if not isinstance(unsorted_list, list):
        return bad_request("Invalid input: unsortedList must be a list of strings.")

    if part == "ONE":
        values: List[int] = []
        for s in unsorted_list:
            if not isinstance(s, str):
                return bad_request("All elements must be strings for part ONE.")
            raw = s.strip()
            v: Optional[int] = None
            if re.fullmatch(r"\d+", raw):
                try:
                    v = int(raw)
                except Exception:
                    v = None
            else:
                v = roman_to_int(raw)
            if v is None:
                return bad_request("Unable to parse number.", details={"value": s})
            values.append(v)
        values.sort()
        return jsonify({"sortedList": [str(x) for x in values]}), 200

    # part TWO
    ranked_items: List[Tuple[int, str, str, int]] = []  # (value, rep, original, index)
    for idx, s in enumerate(unsorted_list):
        if not isinstance(s, str):
            return bad_request("All elements must be strings for part TWO.")
        rep, val = classify_representation(s)
        if val is None:
            return bad_request("Unable to parse number.", details={"value": s})
        ranked_items.append((val, rep, s, idx))

    rep_rank = {
        'ROMAN': 0,
        'ENGLISH': 1,
        'TRAD_CH': 2,
        'SIMP_CH': 3,
        'GERMAN': 4,
        'ARABIC': 5
    }

    ranked_items.sort(key=lambda x: (x[0], rep_rank.get(x[1], 6), x[3]))
    return jsonify({"sortedList": [it[2] for it in ranked_items]}), 200
   

# === THE INK ARCHIVE ===

def _ink_build_graph(goods: List[str], rates: List[Any]):
    n = len(goods)
    best_rate: Dict[Tuple[int, int], float] = {}
    for entry in rates if isinstance(rates, list) else []:
        if not isinstance(entry, (list, tuple)) or len(entry) != 3:
            continue
        try:
            u = int(entry[0])
            v = int(entry[1])
            r = float(entry[2])
        except Exception:
            continue
        if not (0 <= u < n and 0 <= v < n):
            continue
        if r <= 0:
            continue
        key = (u, v)
        if key not in best_rate or r > best_rate[key]:
            best_rate[key] = r
    # edges: (u, v, weight=-log(r), rate=r)
    edges: List[Tuple[int, int, float, float]] = []
    for (u, v), r in best_rate.items():
        try:
            w = -math.log(r)
        except Exception:
            continue
        edges.append((u, v, w, r))
    return n, edges, best_rate


def _ink_product_and_gain_percent(path_idx: List[int], rate_map: Dict[Tuple[int, int], float]) -> Tuple[float, float]:
    if not path_idx or len(path_idx) < 2:
        return 1.0, 0.0
    prod = 1.0
    for i in range(len(path_idx) - 1):
        u = path_idx[i]
        v = path_idx[i + 1]
        r = rate_map.get((u, v))
        if r is None or r <= 0:
            return 0.0, 0.0
        prod *= r
    gain_pct = (prod - 1.0) * 100.0
    return prod, gain_pct


def _ink_bellman_ford_any_cycle(n: int, edges: List[Tuple[int, int, float, float]]) -> Optional[List[int]]:
    # dist initialized to 0 is equivalent to a super source with 0-weight edges
    dist = [0.0] * n
    pred = [-1] * n
    eps = 1e-15

    # Relax edges N-1 times
    for _ in range(max(0, n - 1)):
        changed = False
        for u, v, w, _ in edges:
            if dist[u] + w < dist[v] - eps:
                dist[v] = dist[u] + w
                pred[v] = u
                changed = True
        if not changed:
            break

    # Check for negative cycle
    neg_vertex = -1
    for u, v, w, _ in edges:
        if dist[u] + w < dist[v] - eps:
            neg_vertex = v
            break
    if neg_vertex == -1:
        return None

    # Move inside the cycle
    x = neg_vertex
    for _ in range(n):
        x = pred[x] if x != -1 else x
    # Extract cycle by walking predecessors until repeat
    seen: Dict[int, int] = {}
    order: List[int] = []
    cur = x
    while cur not in seen and cur != -1:
        seen[cur] = len(order)
        order.append(cur)
        cur = pred[cur]
    if cur == -1:
        return None
    start_idx = seen.get(cur, 0)
    cyc = order[start_idx:]
    # Reverse to get forward trading direction and close the loop
    cyc = list(reversed(cyc))
    cyc.append(cyc[0])
    return cyc


def _ink_max_gain_cycle(n: int, edges: List[Tuple[int, int, float, float]], rate_map: Dict[Tuple[int, int], float]) -> Tuple[Optional[List[int]], float]:
    # DP over path length up to N to capture any simple cycle
    best_gain = -1e18
    best_cycle: Optional[List[int]] = None
    eps = 1e-12

    # Pre-extract adjacency for reconstruction convenience (not strictly necessary)
    for s in range(n):
        # dp[k][v]: min total weight to reach v using exactly k edges starting from s
        dp = [[float('inf')] * n for _ in range(n + 1)]
        pred = [[-1] * n for _ in range(n + 1)]
        dp[0][s] = 0.0

        for k in range(1, n + 1):
            for u, v, w, _ in edges:
                if dp[k - 1][u] + w < dp[k][v] - 1e-15:
                    dp[k][v] = dp[k - 1][u] + w
                    pred[k][v] = u

            # If we returned to s with a negative total weight, reconstruct cycle
            if dp[k][s] < -eps:
                # Reconstruct path of length k ending at s
                seq: List[int] = []
                v = s
                kk = k
                for _ in range(k):
                    seq.append(v)
                    v = pred[kk][v]
                    if v == -1:
                        break
                    kk -= 1
                seq.append(v)
                seq.reverse()
                # Trim to the last cycle at s
                try:
                    last_idx = len(seq) - 1
                    first_s = max(i for i in range(0, last_idx) if seq[i] == s)
                    cycle_idx = seq[first_s:last_idx + 1]
                except ValueError:
                    # Fallback: try to isolate a cycle by last repeated node
                    seen_pos: Dict[int, int] = {}
                    cut_i, cut_j = 0, len(seq) - 1
                    for i, node in enumerate(seq):
                        if node in seen_pos:
                            cut_i = seen_pos[node]
                            cut_j = i
                        else:
                            seen_pos[node] = i
                    cycle_idx = seq[cut_i:cut_j + 1]

                # Ensure closed loop
                if cycle_idx[0] != cycle_idx[-1]:
                    cycle_idx.append(cycle_idx[0])

                prod, gain_pct = _ink_product_and_gain_percent(cycle_idx, rate_map)
                gain = gain_pct  # already *100
                if prod > 0 and (gain > best_gain + 1e-9 or (abs(gain - best_gain) <= 1e-9 and (best_cycle is None or len(cycle_idx) < len(best_cycle)))):
                    best_gain = gain
                    best_cycle = cycle_idx

    if best_cycle is None:
        return None, 0.0
    return best_cycle, best_gain


@app.route("/The-Ink-Archive", methods=["POST"]) 
def the_ink_archive():
    payload = request.get_json(silent=True)
    if payload is None or not isinstance(payload, list) or len(payload) == 0:
        return bad_request("Expected a JSON array with two items for Part I and Part II.")

    results: List[Dict[str, Any]] = []

    # Part I: detect any profitable loop
    part1 = payload[0] if len(payload) >= 1 else {}
    goods1 = part1.get("goods", []) if isinstance(part1, dict) else []
    rates1 = part1.get("rates", []) if isinstance(part1, dict) else []
    n1, edges1, rate_map1 = _ink_build_graph(goods1 if isinstance(goods1, list) else [], rates1)

    path1_idx = _ink_bellman_ford_any_cycle(n1, edges1) if n1 > 0 else None
    if path1_idx:
        prod1, gain1 = _ink_product_and_gain_percent(path1_idx, rate_map1)
        path1_names = [goods1[i] for i in path1_idx]
        results.append({"path": path1_names, "gain": gain1})
    else:
        results.append({"path": [], "gain": 0.0})

    # Part II: find the maximum gain cycle
    part2 = payload[1] if len(payload) >= 2 else {}
    goods2 = part2.get("goods", []) if isinstance(part2, dict) else []
    rates2 = part2.get("rates", []) if isinstance(part2, dict) else []
    n2, edges2, rate_map2 = _ink_build_graph(goods2 if isinstance(goods2, list) else [], rates2)

    if n2 > 0:
        best_cycle2, best_gain2 = _ink_max_gain_cycle(n2, edges2, rate_map2)
        if best_cycle2:
            path2_names = [goods2[i] for i in best_cycle2]
            results.append({"path": path2_names, "gain": best_gain2})
        else:
            results.append({"path": [], "gain": 0.0})
    else:
        results.append({"path": [], "gain": 0.0})

    resp = make_response(jsonify(results), 200)
    resp.headers["Content-Type"] = "application/json"
    return resp

if __name__ == "__main__":
    # For local development only
    # app.run()
    app.run(host='0.0.0.0', port=3000, debug=False)
