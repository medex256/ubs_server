from flask import Flask, request, jsonify, make_response
from math import hypot
from typing import Any, Dict, List, Tuple, Optional, Set
from scipy.stats import linregress
from scipy import interpolate
import numpy as np
from collections import defaultdict, deque
import re, math

app = Flask(__name__)


def bad_request(message: str, details: Optional[Dict[str, Any]] = None, status_code: int = 400):
    payload = {"error": message}
    if details:
        payload["details"] = details
    resp = make_response(jsonify(payload), status_code)
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
            4,  # "Trivia!": How many challenges are there this year, which title ends with an exclamation mark?
            1,  # "Ticketing Agent": What type of tickets is the ticketing agent handling?
            2,  # "Blankety Blanks": How many lists and elements per list are included in the dataset you must impute?
            2,  # "Princess Diaries": What's Princess Mia's cat name in the movie Princess Diaries?
            3,  # "MST Calculation": What is the average number of nodes in a test case?
            4,  # "Universal Bureau of Surveillance": Which singer did not have a James Bond theme song?
            3,  # "Operation Safeguard": What is the smallest font size in our question?
            4,  # "Capture The Flag": Which of these are anagrams of the challenge name?
            4   # "Filler 1": Where has UBS Global Coding Challenge been held before?
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


def find_extra_channels(network: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Find extra channels that can be safely removed from a spy network.
    The goal is to find ALL edges that can be individually removed while maintaining connectivity.
    """
    if not network:
        return []
    
    # Build adjacency list and edge list
    graph = defaultdict(set)
    edges = []
    
    for connection in network:
        spy1 = connection.get("spy1")
        spy2 = connection.get("spy2")
        if spy1 and spy2 and spy1 != spy2:
            graph[spy1].add(spy2)
            graph[spy2].add(spy1)
            edges.append((spy1, spy2))
    
    if not graph:
        return []
    
    # Find all nodes
    all_nodes = set(graph.keys())
    n_nodes = len(all_nodes)
    
    # We need at least (n_nodes - 1) edges to maintain connectivity
    min_edges_needed = n_nodes - 1
    
    if len(edges) <= min_edges_needed:
        return []  # No extra edges to remove
    
    def is_connected_without_edge(edges_to_test: List[Tuple[str, str]], exclude_edge: Tuple[str, str]) -> bool:
        """Check if the graph remains connected when excluding a specific edge."""
        # Build graph without the excluded edge
        test_graph = defaultdict(set)
        for spy1, spy2 in edges_to_test:
            if (spy1, spy2) != exclude_edge and (spy2, spy1) != exclude_edge:
                test_graph[spy1].add(spy2)
                test_graph[spy2].add(spy1)
        
        if not test_graph:
            return False
        
        # BFS to check connectivity
        start_node = next(iter(test_graph.keys()))
        visited = set()
        queue = deque([start_node])
        visited.add(start_node)
        
        while queue:
            node = queue.popleft()
            for neighbor in test_graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return len(visited) == n_nodes
    
    # Find all edges that can be safely removed
    extra_channels = []
    for spy1, spy2 in edges:
        if is_connected_without_edge(edges, (spy1, spy2)):
            extra_channels.append({"spy1": spy1, "spy2": spy2})
    
    return extra_channels


@app.route("/investigate", methods=["POST"])
def investigate():
    """
    Analyze spy networks to find extra channels that can be safely removed.
    """
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"networks": []}), 200
    
    networks_data = data.get("networks", [])
    if not isinstance(networks_data, list):
        return jsonify({"networks": []}), 200
    
    result_networks = []
    
    for network_data in networks_data:
        if not isinstance(network_data, dict):
            continue
            
        network_id = network_data.get("networkId")
        network = network_data.get("network", [])
        
        if not network_id or not isinstance(network, list):
            continue
        
        # Find extra channels for this network
        extra_channels = find_extra_channels(network)
        
        result_networks.append({
            "networkId": network_id,
            "extraChannels": extra_channels
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
        # Robust conversion of a LaTeX-like expression to a safe Python expression.
        s = expr or ''
        s = replace_text_commands(s)
        s = replace_times_dot(s)
        s = remove_latex_wrappers(s)
        s = replace_frac(s)
        s = replace_exp_e(s)
        s = replace_summation(s)
        s = replace_latex_commands(s)
        s = replace_subscripts_and_brackets(s)
        s = replace_superscripts(s)

        # Remove $ markers and normalize braces
        s = s.replace('$', '')
        s = s.replace('{', '(').replace('}', ')')

        # Normalize unicode minus signs to ASCII
        s = s.replace('\u2212', '-')
        s = s.replace('−', '-')

        # Remove thousands separators: 1,000 -> 1000
        s = re.sub(r'(?<=\d),(?=\d)', '', s)

        # Convert percentages: 3% -> (3/100)
        s = re.sub(r'(?P<num>\d+(?:\.\d+)?)\s*(?:\\%|%)', lambda m: f'(({m.group("num")})/100)', s)

        # Absolute value bars: |x| -> abs(x)
        prev = None
        while prev != s:
            prev = s
            s = re.sub(r'\|([^|]+)\|', r'abs(\1)', s)

        # Collapse whitespace
        s = re.sub(r'\s+', ' ', s).strip()

        # Implied multiplication insertion but avoid breaking function calls like max( or sin(
        known_funcs = {
            'max','min','sum','exp','log','log10','pow','abs','round',
            'sin','cos','tan','sqrt','floor','ceil'
        }

        def _func_replace(m):
            name = m.group(1)
            # If this looks like a known function call (no space originally), do not insert '*'
            if name in known_funcs:
                return name + '('
            return name + '*('

        # Insert '*' when identifier is followed by SPACE then '(': 'x (' -> 'x*('
        s = re.sub(r'([A-Za-z_][A-Za-z0-9_]*)\s+\(', _func_replace, s)
        # Insert '*' when number or ')' directly before '(': '2(' or ')(' -> '2*(' or ')*('
        s = re.sub(r'(?<=[0-9\)])\s*(?=\()', '*', s)
        # Insert '*' between number and identifier: '2x' -> '2*x'
        s = re.sub(r'(?<=[0-9\.])\s*(?=[A-Za-z_])', '*', s)
        # Insert '*' between ')' and identifier/number: ')x' -> ')*x'
        s = re.sub(r'(?<=\))\s*(?=[A-Za-z_0-9])', '*', s)

        return s

    # Expanded safe functions and constants
    safe_funcs = {
        'exp': math.exp,
        'log': math.log,
        'log10': math.log10,
        'max': max,
        'min': min,
        'sum': sum,
        'pow': pow,
        'abs': abs,
        'round': round,
        'sqrt': math.sqrt,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'floor': math.floor,
        'ceil': math.ceil,
        'pi': math.pi,
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

            # Prepare local environment with provided variables
            local_env = {}
            for k, v in variables.items():
                key = str(k)
                # normalize numeric-like strings
                try:
                    val = float(v)
                except Exception:
                    val = v
                # base key
                if key not in local_env:
                    local_env[key] = val
                # sanitized variants
                vk = key.replace('[', '_').replace(']', '').replace(' ', '_')
                vk2 = re.sub(r'[^0-9A-Za-z_]', '_', key)
                vk3 = vk2.lower()
                for sk in (vk, vk2, vk3):
                    if sk and sk not in local_env:
                        local_env[sk] = val

            # Merge safe funcs without overwriting variables
            for fname, fobj in safe_funcs.items():
                if fname not in local_env:
                    local_env[fname] = fobj

            # Evaluate expression in restricted environment
            value = eval(py, {'__builtins__': None}, local_env)
            results.append({'result': round(float(value), 4)})
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

def _integrate_scan(state: Dict[str, Any], crow_id: str, scan_result: List[List[str]]):
    n = state["length"]
    cx, cy = state["crows"][crow_id]

    # The scan_result is 5x5 centered at crow, rows top-to-bottom, cols left-to-right
    for r in range(5):
        for c in range(5):
            symbol = scan_result[r][c]
            x = cx + (c - 2)
            y = cy + (r - 2)
            if not _in_bounds(x, y, n):
                continue
            if symbol == "X":
                continue
            if symbol == "C" or symbol == "*":
                state["known_empty"].add((x, y))
            elif symbol == "W":
                state["known_walls"].add((x, y))
    # Always ensure the crow's current cell is empty
    state["known_empty"].add((cx, cy))

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

    # Dynamic scan threshold based on remaining unknown area
    if unknown_cells > 0.6 * total_cells:
        scan_threshold = 10
    elif unknown_cells > 0.3 * total_cells:
        scan_threshold = 6
    else:
        scan_threshold = 3

    # 1) Consider scanning now from any crow with high expected gain
    best_scan = None  # (gain, crow_id)
    for crow_id, (cx, cy) in state["crows"].items():
        gain = _scan_unknown_count_at((cx, cy), n, state["known_empty"], state["known_walls"])
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
    for crow_id, (sx, sy) in state["crows"].items():
        if (sx, sy) not in state["known_empty"]:
            # Ensure current crow cell is marked empty
            state["known_empty"].add((sx, sy))

        first_dir, dist = _bfs_first_step_direction(state, (sx, sy))
        # Evaluate all reachable known-empty cells as potential scan centers
        for cell, d in dist.items():
            gain = _scan_unknown_count_at(cell, n, state["known_empty"], state["known_walls"])
            if gain <= 0:
                continue
            direction = first_dir.get(cell)
            if not direction:
                continue
            # Prefer higher gain, then shorter distance
            score = (gain, -d)
            if best_move is None or score > best_move[0]:
                best_move = (score, crow_id, direction)

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
            if best_probe is None or gain > best_probe[0]:
                best_probe = (gain, crow_id, direction)

    if best_probe is not None:
        _, crow_id, direction = best_probe
        return {
            "action_type": "move",
            "crow_id": crow_id,
            "direction": direction,
        }

    # 4) Fallback: if absolutely nothing to do, scan with any crow (should be rare)
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
        }
        for c in crows_list:
            cid = str(c.get("id"))
            x = int(c.get("x", 0))
            y = int(c.get("y", 0))
            state["crows"][cid] = (x, y)
            if _in_bounds(x, y, length_of_grid):
                state["known_empty"].add((x, y))
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

    resp = make_response(jsonify(response), 200)
    resp.headers["Content-Type"] = "application/json"
    return resp
   
if __name__ == "__main__":
    # For local development only
    app.run()
