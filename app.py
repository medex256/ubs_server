from flask import Flask, request, jsonify, make_response
from math import hypot
from typing import Any, Dict, List, Tuple, Optional
from scipy.stats import linregress
from scipy import interpolate
import numpy as np

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
    
    tasks = data.get("tasks", [])
    subway = data.get("subway", [])
    starting_station = data.get("starting_station")
    
    if not isinstance(tasks, list) or not isinstance(subway, list) or starting_station is None:
        return bad_request("Invalid input format.")
    if not tasks:
        return jsonify({"max_score": 0, "min_fee": 0, "schedule": []}), 200
    
    # ------------------------------------------------------------------
    # 1. Build station mapping and distance matrix (FAST)
    # ------------------------------------------------------------------
    stations = {starting_station}
    for t in tasks:
        stations.add(t["station"])
    for r in subway:
        if len(r.get("connection", [])) == 2:
            stations.update(r["connection"])
    
    idx_of = {s: i for i, s in enumerate(stations)}
    n_st = len(stations)
    
    inf = float("inf")
    dist = [[inf] * n_st for _ in range(n_st)]
    for i in range(n_st):
        dist[i][i] = 0
    
    for r in subway:
        u, v = r.get("connection", [None, None])
        fee = r.get("fee", 0)
        if u in idx_of and v in idx_of:
            iu, iv = idx_of[u], idx_of[v]
            if fee < dist[iu][iv]:
                dist[iu][iv] = dist[iv][iu] = fee
    
    # Floyd-Warshall
    for k in range(n_st):
        for i in range(n_st):
            for j in range(n_st):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    
    # ------------------------------------------------------------------
    # 2. Sort tasks and find compatible successors
    # ------------------------------------------------------------------
    tasks.sort(key=lambda t: t["start"])
    n = len(tasks)
    
    # Pre-compute all compatible successors for each task
    compatible = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if tasks[j]["start"] >= tasks[i]["end"]:
                compatible[i].append(j)
    
    s_idx = idx_of[starting_station]
    
    # ------------------------------------------------------------------
    # 3. Corrected iterative DP - explore ALL paths
    # ------------------------------------------------------------------
    # dp[i] = (max_score, min_fee, next_task) for optimal solution starting from task i
    dp_score = [0] * (n + 1)  # n is sentinel for "end"
    dp_fee = [0] * (n + 1)
    dp_next = [n] * (n + 1)
    
    for i in range(n - 1, -1, -1):
        # Option 1: Skip task i
        best_score = dp_score[i + 1]
        best_fee = dp_fee[i + 1]
        best_next = i + 1  # skip to next task
        
        # Option 2: Take task i and try all compatible successors
        ui = idx_of[tasks[i]["station"]]
        curr_score = tasks[i]["score"]
        
        # Try ending here (go directly to starting station)
        end_fee = dist[ui][s_idx] + dp_fee[n]  # return home
        total_score = curr_score + dp_score[n]
        total_fee = end_fee
        
        if total_score > best_score or (total_score == best_score and total_fee < best_fee):
            best_score, best_fee, best_next = total_score, total_fee, n
        
        # Try each compatible successor
        for j in compatible[i]:
            uj = idx_of[tasks[j]["station"]]
            travel_fee = dist[ui][uj]  # cost to go to next task
            
            successor_score = dp_score[j]
            successor_fee = dp_fee[j]
            
            total_score = curr_score + successor_score
            total_fee = travel_fee + successor_fee
            
            if total_score > best_score or (total_score == best_score and total_fee < best_fee):
                best_score, best_fee, best_next = total_score, total_fee, j
        
        dp_score[i] = best_score
        dp_fee[i] = best_fee
        dp_next[i] = best_next
    
    # ------------------------------------------------------------------
    # 4. Find best starting task (including initial travel cost)
    # ------------------------------------------------------------------
    max_score = 0
    min_fee = 0
    best_start = n  # no tasks
    
    # Option: Take no tasks
    if best_start == n:
        pass  # already set above
    
    # Try starting with each task
    for i in range(n):
        ui = idx_of[tasks[i]["station"]]
        initial_travel = dist[s_idx][ui]  # cost from starting station to first task
        
        total_score = dp_score[i]
        total_fee = initial_travel + dp_fee[i]
        
        if total_score > max_score or (total_score == max_score and total_fee < min_fee):
            max_score, min_fee, best_start = total_score, total_fee, i
    
    # ------------------------------------------------------------------
    # 5. Reconstruct schedule
    # ------------------------------------------------------------------
    schedule = []
    current = best_start
    while current < n:
        schedule.append(tasks[current]["name"])
        current = dp_next[current]
    
    response = {
        "max_score": max_score,
        "min_fee": min_fee,
        "schedule": schedule
    }
    return jsonify(response), 200


   
if __name__ == "__main__":
    # For local development only
    app.run()
