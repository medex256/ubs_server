from flask import Flask, request, jsonify, make_response
from math import hypot
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


@app.route("/ticketing-agent", methods=["POST"])
def ticketing_agent():
    content_type = request.headers.get("Content-Type", "")
    if not content_type.lower().startswith("application/json"):
        return bad_request("Unsupported Media Type. Use Content-Type: application/json.", status_code=415)

    data = request.get_json(silent=True)
    if data is None:
        return bad_request("Invalid or malformed JSON body.")


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
            # Skip invalid concert entries
            continue
        concert_by_name[name] = c
        concert_locations[name] = loc

    assignments: List[Dict[str, Any]] = []
    unassigned: List[Dict[str, Any]] = []

    for cust in customers:
        cname = cust.get("name") or ""
        vip = parse_bool(cust.get("vip_status"))
        cloc = as_xy(cust.get("location"))
        card = cust.get("credit_card")

        reason = None
        chosen_concert = None

        # Priority by credit card mapping
        if isinstance(card, str) and card in priority_map:
            preferred = priority_map.get(card)
            if isinstance(preferred, str) and preferred in concert_by_name:
                chosen_concert = preferred
                reason = f"priority({card})"

        # Otherwise pick nearest booking center
        if chosen_concert is None and cloc is not None and concert_locations:
            cx, cy = cloc
            best_name = None
            best_dist = None
            for name, (x, y) in concert_locations.items():
                d = hypot(cx - x, cy - y)
                if best_dist is None or d < best_dist:
                    best_dist = d
                    best_name = name
            if best_name is not None:
                chosen_concert = best_name
                reason = "nearest"

        record = {
            "customer": cname,
            "vip_status": vip,
            "credit_card": card,
        }

        if chosen_concert:
            record.update({
                "concert": chosen_concert,
                "reason": reason
            })
            assignments.append(record)
        else:
            record.update({"error": "no_concert_available_or_invalid_location"})
            unassigned.append(record)

    response_body = {
        "assignments": assignments,
        "unassigned": unassigned,
        "stats": {
            "customers": len(customers),
            "concerts": len(concert_by_name),
            "assigned": len(assignments),
            "unassigned": len(unassigned)
        }
    }

    resp = make_response(jsonify(response_body), 200)
    resp.headers["Content-Type"] = "application/json"
    return resp


@app.route("/")
def root():
    resp = make_response(jsonify({"service": "ticketing-agent", "status": "ok"}), 200)
    resp.headers["Content-Type"] = "application/json"
    return resp


if __name__ == "__main__":
    # For local development only
    app.run(host="0.0.0.0", port=5000)
