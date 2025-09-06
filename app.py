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
    np_series = np.array([float(x) if x is not None else np.nan for x in series])
    imputed = linear_interpolation_np(np_series)
    return imputed.tolist()


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
    
    # Preprocess tasks: sort by start time
    tasks_sorted = sorted(tasks, key=lambda t: t["start"])
    
    # Build subway graph - find all unique stations
    stations = set()
    for route in subway:
        conn = route.get("connection", [])
        if len(conn) == 2:
            stations.add(conn[0])
            stations.add(conn[1])
    stations.add(starting_station)
    
    # Map station IDs to consecutive indices for efficient matrix operations
    station_map = {s: i for i, s in enumerate(sorted(stations))}
    num_stations = len(stations)
    
    # Initialize distance matrix with infinity
    inf = float('inf')
    distances = [[inf for _ in range(num_stations)] for _ in range(num_stations)]
    
    # Set diagonal to 0 (distance to self)
    for i in range(num_stations):
        distances[i][i] = 0
    
    # Fill in direct connections
    for route in subway:
        conn = route.get("connection", [])
        fee = route.get("fee", 0)
        if len(conn) == 2:
            i = station_map[conn[0]]
            j = station_map[conn[1]]
            distances[i][j] = fee
            distances[j][i] = fee  # Undirected graph
    
    # Floyd-Warshall algorithm for all-pairs shortest paths
    for k in range(num_stations):
        for i in range(num_stations):
            for j in range(num_stations):
                if distances[i][k] != inf and distances[k][j] != inf:
                    distances[i][j] = min(distances[i][j], distances[i][k] + distances[k][j])
    
    # Convert starting station to index
    starting_idx = station_map[starting_station]
    
    # Dynamic Programming to find optimal schedule
    memo = {}
    
    def dp(index, last_task_idx):
        # Base case: no more tasks to consider
        if index == len(tasks_sorted):
            # Calculate fee to return to starting station
            return_fee = 0
            if last_task_idx != -1:
                last_station = tasks_sorted[last_task_idx]["station"]
                last_idx = station_map[last_station]
                return_fee = distances[last_idx][starting_idx]
            return (0, return_fee, [])
        
        # Check memoization table
        key = (index, last_task_idx)
        if key in memo:
            return memo[key]
        
        # Option 1: Skip current task
        skip_score, skip_fee, skip_schedule = dp(index + 1, last_task_idx)
        
        # Option 2: Take current task if possible
        curr_task = tasks_sorted[index]
        can_take = True
        
        # Check for time overlap with previous task
        if last_task_idx != -1:
            last_task = tasks_sorted[last_task_idx]
            if last_task["end"] > curr_task["start"]:
                can_take = False
        
        take_score, take_fee, take_schedule = (0, float('inf'), [])
        
        if can_take:
            # Calculate travel fee to current task
            travel_fee = 0
            curr_idx = station_map[curr_task["station"]]
            
            if last_task_idx == -1:  # Coming from starting station
                travel_fee = distances[starting_idx][curr_idx]
            else:  # Coming from previous task
                prev_station = tasks_sorted[last_task_idx]["station"]
                prev_idx = station_map[prev_station]
                travel_fee = distances[prev_idx][curr_idx]
            
            # Recursive call to find best schedule after taking this task
            next_score, next_fee, next_schedule = dp(index + 1, index)
            
            take_score = curr_task["score"] + next_score
            take_fee = travel_fee + next_fee
            take_schedule = [curr_task["name"]] + next_schedule
        
        # Choose the better option (score first, fee second)
        if take_score > skip_score or (take_score == skip_score and take_fee < skip_fee):
            result = (take_score, take_fee, take_schedule)
        else:
            result = (skip_score, skip_fee, skip_schedule)
        
        # Memoize and return
        memo[key] = result
        return result
    
    # Run the DP algorithm
    max_score, min_fee, schedule = dp(0, -1)
    
    # Prepare response
    response = {
        "max_score": max_score,
        "min_fee": min_fee,
        "schedule": schedule
    }
    
    resp = make_response(jsonify(response), 200)
    resp.headers["Content-Type"] = "application/json"
    return resp

if __name__ == "__main__":
    # For local development only
    app.run()
