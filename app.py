from flask import Flask, request, jsonify, make_response
from math import hypot
import os
from typing import Any, Dict, List, Tuple, Optional

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


@app.route("/trivia", methods=["POST"])
def trivia():
    res = {
        "answers": [4, 1, 2]
    }
    return res

@app.route("/ticketing-agent", methods=["POST"])
def ticketing_agent():
    data = request.get_json(silent=True)
    if data is None:
        return bad_request("Invalid JSON body.")


    customers: List[Dict[str, Any]] = data.get("customers", [])
    concerts: List[Dict[str, Any]] = data.get("concerts", [])
    priority_map: Dict[str, str] = data.get("priority", {})

    # Index concerts by name and pre-parse locations
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

if __name__ == "__main__":
    # For local development only
    app.run()
