from flask import Flask, request, jsonify, make_response
from math import hypot
from typing import Any, Dict, List, Tuple, Optional
from scipy.stats import linregress
from scipy import interpolate
import numpy as np
import re
import math

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


@app.route("/")
def root():
    return "OK", 200


@app.route("/princess-diaries", methods=["POST"])
def princess_diaries():
    data = request.get_json(silent=True)
    if not data:
        return bad_request("Invalid JSON body.")
    
    # Extract data
    tasks = data.get("tasks", [])
    subway = data.get("subway", [])
    starting_station = data.get("starting_station")
    
    if not isinstance(tasks, list) or not isinstance(subway, list) or starting_station is None:
        return bad_request("Invalid input format.")
    
    # Early exit for empty case
    if not tasks:
        return jsonify({"max_score": 0, "min_fee": 0, "schedule": []}), 200
    
    # Build adjacency list instead of matrix (more memory efficient)
    graph = {}
    for route in subway:
        conn = route.get("connection", [])
        fee = route.get("fee", 0)
        if len(conn) == 2:
            u, v = conn
            if u not in graph: graph[u] = {}
            if v not in graph: graph[v] = {}
            graph[u][v] = fee
            graph[v][u] = fee  # Undirected graph
    
    # Ensure starting station exists in graph
    if starting_station not in graph:
        graph[starting_station] = {}
    
    # Distance cache to avoid repeated calculations
    distance_cache = {}
    
    def get_shortest_path(start, end):
        # Check cache first
        if (start, end) in distance_cache:
            return distance_cache[(start, end)]
        
        # Use Dijkstra's algorithm for single source
        distances = {node: float('inf') for node in graph}
        distances[start] = 0
        
        import heapq
        queue = [(0, start)]
        visited = set()
        
        while queue:
            dist, current = heapq.heappop(queue)
            
            if current == end:
                break
                
            if current in visited:
                continue
            
            visited.add(current)
            
            for neighbor, weight in graph[current].items():
                if neighbor not in visited:
                    new_dist = dist + weight
                    if new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        heapq.heappush(queue, (new_dist, neighbor))
        
        # Cache all computed distances
        for node, dist in distances.items():
            distance_cache[(start, node)] = dist
        
        return distances[end]
    
    # Sort tasks by start time
    tasks_sorted = sorted(tasks, key=lambda t: t["start"])
    
    # Pre-compute next compatible tasks for each task (huge speedup)
    n = len(tasks_sorted)
    next_compatible = [[] for _ in range(n)]
    
    for i in range(n):
        end_time = tasks_sorted[i]["end"]
        for j in range(i+1, n):
            if tasks_sorted[j]["start"] >= end_time:
                next_compatible[i].append(j)
                # Only need first few compatible tasks
                if len(next_compatible[i]) >= 10:
                    break
    
    # Compute max possible score for remaining tasks (for pruning)
    max_remaining = [0] * (n + 1)
    for i in range(n-1, -1, -1):
        max_remaining[i] = tasks_sorted[i]["score"] + max_remaining[i+1]
    
    # Dynamic Programming with efficient memoization
    best_score_found = 0
    memo = {}
    
    def dp(task_idx, prev_task_idx):
        # Base case: no more tasks to consider
        if task_idx == n:
            # Calculate fee to return to starting station
            return_fee = 0
            if prev_task_idx != -1:
                prev_station = tasks_sorted[prev_task_idx]["station"]
                return_fee = get_shortest_path(prev_station, starting_station)
            return (0, return_fee, [])
        
        # Early termination: if max possible remaining score can't beat best found
        nonlocal best_score_found
        if prev_task_idx != -1:
            current_score = sum(tasks_sorted[i]["score"] for i in range(prev_task_idx+1))
            if current_score + max_remaining[task_idx] <= best_score_found:
                return (0, float('inf'), [])
        
        # Check memoization table
        key = (task_idx, prev_task_idx)
        if key in memo:
            return memo[key]
        
        # Option 1: Skip current task
        skip_score, skip_fee, skip_schedule = dp(task_idx + 1, prev_task_idx)
        
        # Option 2: Take current task if possible
        curr_task = tasks_sorted[task_idx]
        can_take = True
        
        # Check for time overlap with previous task
        if prev_task_idx != -1:
            prev_task = tasks_sorted[prev_task_idx]
            if prev_task["end"] > curr_task["start"]:
                can_take = False
        
        if can_take:
            # Calculate travel fee to current task
            curr_station = curr_task["station"]
            
            if prev_task_idx == -1:  # Coming from starting station
                travel_fee = get_shortest_path(starting_station, curr_station)
            else:  # Coming from previous task
                prev_station = tasks_sorted[prev_task_idx]["station"]
                travel_fee = get_shortest_path(prev_station, curr_station)
            
            # Find next compatible tasks (much more efficient than checking all)
            next_idx = n
            for j in next_compatible[task_idx]:
                next_idx = j
                break
                
            # Recursive call with next compatible task or end
            next_score, next_fee, next_schedule = dp(next_idx, task_idx)
            
            take_score = curr_task["score"] + next_score
            take_fee = travel_fee + next_fee
            take_schedule = [curr_task["name"]] + next_schedule
            
            # Update best score found (for pruning)
            best_score_found = max(best_score_found, take_score)
        else:
            take_score, take_fee, take_schedule = 0, float('inf'), []
        
        # Choose the better option (score first, fee second)
        if take_score > skip_score or (take_score == skip_score and take_fee < skip_fee):
            result = (take_score, take_fee, take_schedule)
        else:
            result = (skip_score, skip_fee, skip_schedule)
        
        # Memoize and return
        memo[key] = result
        return result
    
    # Run the optimized DP algorithm
    max_score, min_fee, schedule = dp(0, -1)
    
    # Prepare response
    response = {
        "max_score": max_score,
        "min_fee": min_fee,
        "schedule": schedule
    }
    
    return jsonify(response), 200

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

if __name__ == "__main__":
    # For local development only
    app.run()
