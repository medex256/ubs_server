from flask import Flask, request, jsonify, make_response
from math import hypot
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
from scipy import interpolate, signal

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


class TimeSeriesImputer:
    """Robust time series imputation using multiple methods and ensemble approach."""
    
    def __init__(self):
        self.methods = [
            'linear_interpolation',
            'cubic_spline',
            'polynomial_trend',
            'autoregressive',
            'seasonal_decomposition',
            'gaussian_process'
        ]
    
    def linear_interpolation(self, series: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if np.sum(~mask) < 2:
            return series
        valid_indices = np.where(~mask)[0]
        valid_values = series[~mask]
        return np.interp(np.arange(len(series)), valid_indices, valid_values)
    
    def cubic_spline(self, series: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if np.sum(~mask) < 4:
            return self.linear_interpolation(series, mask)
        valid_indices = np.where(~mask)[0]
        valid_values = series[~mask]
        try:
            spline = interpolate.CubicSpline(valid_indices, valid_values, bc_type='natural')
            result = spline(np.arange(len(series)))
            return np.clip(result, -1e6, 1e6) 
        except:
            return self.linear_interpolation(series, mask)
    
    def polynomial_trend(self, series: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if np.sum(~mask) < 3:
            return self.linear_interpolation(series, mask)
        
        valid_indices = np.where(~mask)[0]
        valid_values = series[~mask]
        
        best_score = float('inf')
        best_result = series.copy()
        
        for degree in [1, 2, 3]:
            try:
                poly = np.polyfit(valid_indices, valid_values, degree)
                result = np.polyval(poly, np.arange(len(series)))
                if len(valid_values) > 5:
                    cv_score = np.mean([np.mean((valid_values[i::3] - np.polyval(poly, valid_indices[i::3]))**2) 
                                      for i in range(3)])
                    if cv_score < best_score:
                        best_score = cv_score
                        best_result = np.clip(result, -1e6, 1e6)
            except:
                continue
        
        return best_result
    
    def autoregressive(self, series: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if np.sum(~mask) < 10:
            return self.linear_interpolation(series, mask)
        
        result = series.copy()
        valid_data = series[~mask]
        
        if len(valid_data) > 1:
            ar_coef = np.corrcoef(valid_data[:-1], valid_data[1:])[0, 1]
            ar_coef = np.clip(ar_coef, -0.99, 0.99)
            
            for i in range(len(series)):
                if mask[i]:
                    if i == 0:
                        result[i] = np.mean(valid_data)
                    else:
                        result[i] = ar_coef * result[i-1] + (1 - ar_coef) * np.mean(valid_data)
        
        return np.clip(result, -1e6, 1e6)
    
    def seasonal_decomposition(self, series: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if np.sum(~mask) < 20:
            return self.linear_interpolation(series, mask)
        
        result = series.copy()
        valid_indices = np.where(~mask)[0]
        valid_values = series[~mask]
        
        if len(valid_values) > 10:
            autocorr = np.correlate(valid_values - np.mean(valid_values), 
                                  valid_values - np.mean(valid_values), mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            peaks, _ = signal.find_peaks(autocorr[1:min(50, len(autocorr))], height=0.1)
            if len(peaks) > 0:
                period = peaks[0] + 1
                
                seasonal_pattern = np.zeros(period)
                for i in range(period):
                    seasonal_indices = valid_indices[valid_indices % period == i]
                    if len(seasonal_indices) > 0:
                        seasonal_pattern[i] = np.mean(valid_values[valid_indices % period == i])
                
                for i in range(len(series)):
                    if mask[i]:
                        result[i] = seasonal_pattern[i % period]
            else:
                result = self.linear_interpolation(series, mask)
        else:
            result = self.linear_interpolation(series, mask)
        
        return np.clip(result, -1e6, 1e6)
    
    def gaussian_process(self, series: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if np.sum(~mask) < 5:
            return self.linear_interpolation(series, mask)
        
        valid_indices = np.where(~mask)[0]
        valid_values = series[~mask]
        
        result = series.copy()
        for i in range(len(series)):
            if mask[i]:
                distances = np.abs(valid_indices - i)
                weights = np.exp(-distances / (len(series) / 10)) 
                weights = weights / np.sum(weights)
                result[i] = np.sum(weights * valid_values)
        
        return np.clip(result, -1e6, 1e6)
    
    def ensemble_impute(self, series: np.ndarray) -> np.ndarray:
        """Ensemble approach combining multiple imputation methods."""
        mask = np.isnan(series)
        
        if np.sum(~mask) == 0:
            return np.zeros_like(series)
        
        if np.sum(~mask) == 1:
            return np.full_like(series, series[~mask][0])
        
        predictions = []
        for method_name in self.methods:
            method = getattr(self, method_name)
            try:
                pred = method(series.copy(), mask)
                if not np.any(np.isnan(pred)) and not np.any(np.isinf(pred)):
                    predictions.append(pred)
            except:
                continue
        
        if not predictions:
            return self.linear_interpolation(series, mask)
        
        weights = np.ones(len(predictions))
        
        for i, pred in enumerate(predictions):
            smoothness = 1.0 / (1.0 + np.var(np.diff(pred)))
            # variance
            if np.sum(~mask) > 1:
                mse = np.mean((pred[~mask] - series[~mask])**2)
                accuracy = 1.0 / (1.0 + mse)
                weights[i] = smoothness * accuracy
            else:
                weights[i] = smoothness
        
        weights = weights / np.sum(weights)
        
        # Weighted average of predictions
        result = np.zeros_like(series)
        for pred, weight in zip(predictions, weights):
            result += weight * pred
        
        result = np.clip(result, -1e6, 1e6)
        
        # Replace any remaining NaN/Inf values
        result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return result


def impute_series(series: List[Any]) -> List[float]:
    np_series = np.array([float(x) if x is not None else np.nan for x in series])
    
    imputer = TimeSeriesImputer()
    imputed = imputer.ensemble_impute(np_series)
    
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
        
        resp = make_response(jsonify(imputed_series), 200)
        resp.headers["Content-Type"] = "application/json"
        return resp
        
    except Exception as e:
        return bad_request(f"Imputation failed: {str(e)}", status_code=500)


@app.route("/trivia", methods=["GET"])
def trivia():
    res = {
        "answers": [4, 1, 2, ]
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
    resp = make_response(jsonify({"service": "ticketing-agent", "status": "ok"}), 200)
    resp.headers["Content-Type"] = "application/json"
    return resp


if __name__ == "__main__":
    # For local development only
    app.run()
