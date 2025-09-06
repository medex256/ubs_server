from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import numpy as np
import random
import time
import re
import math
from typing import List, Dict, Any, Optional, Tuple

app = Flask(__name__)
CORS(app)

# Set JSON configuration
app.config['JSON_SORT_KEYS'] = False
app.json.sort_keys = False

def bad_request(message: str, details: Optional[Dict[str, Any]] = None, status_code: int = 400):
    """Helper function to return standardized error responses"""
    payload = {"error": message}
    if details:
        payload["details"] = details
    resp = make_response(jsonify(payload), status_code)
    resp.headers["Content-Type"] = "application/json"
    return resp

@app.route("/", methods=["GET"])
def root():
    """Health check endpoint"""
    return jsonify({"status": "OK", "message": "Trading Bot API is running"}), 200

# Trading Bot Core Logic
class TradingBot:
    """Advanced crypto trading bot with sentiment analysis and technical indicators"""
    
    def __init__(self):
        # Sentiment keywords for news analysis
        self.bullish_keywords = [
            'buy', 'bull', 'bullish', 'moon', 'pump', 'rally', 'surge', 'breakout', 'adoption',
            'institutional', 'reserve', 'etf', 'approval', 'partnership', 'growth', 'positive',
            'upgrade', 'support', 'breakthrough', 'milestone', 'innovation', 'strategic',
            'investment', 'accumulation', 'halving', 'scarcity', 'treasury', 'allocation',
            'strength', 'momentum', 'bullmarket', 'uptick', 'gains', 'rise', 'climb'
        ]
        
        self.bearish_keywords = [
            'sell', 'bear', 'bearish', 'dump', 'crash', 'correction', 'drop', 'fall',
            'regulation', 'ban', 'crackdown', 'lawsuit', 'hack', 'security', 'concern',
            'risk', 'volatile', 'uncertainty', 'decline', 'loss', 'liquidation',
            'bubble', 'overvalued', 'skeptical', 'warning', 'caution', 'negative',
            'weakness', 'breakdown', 'resistance', 'profit-taking', 'fear'
        ]
        
        # High impact keywords that amplify sentiment
        self.high_impact_keywords = [
            'trump', 'biden', 'fed', 'federal', 'government', 'sec', 'cftc', 'treasury',
            'blackrock', 'fidelity', 'vanguard', 'grayscale', 'microstrategy', 'tesla',
            'coinbase', 'binance', 'institutional', 'whale', 'massive', 'huge', 'billions'
        ]
    
    def analyze_sentiment(self, title: str, source: str) -> float:
        """Analyze sentiment from news title and source with weighted scoring"""
        title_lower = title.lower()
        
        # Base sentiment score
        bullish_score = sum(1 for word in self.bullish_keywords if word in title_lower)
        bearish_score = sum(1 for word in self.bearish_keywords if word in title_lower)
        
        # Apply high impact multiplier
        impact_multiplier = 1.0
        if any(word in title_lower for word in self.high_impact_keywords):
            impact_multiplier = 2.0
        
        # Source credibility weighting
        source_weight = 1.0
        if source.lower() in ['reuters', 'bloomberg', 'wsj', 'financial times']:
            source_weight = 1.5
        elif source.lower() in ['twitter', 'reddit']:
            source_weight = 0.8
        
        # Calculate net sentiment with weights
        net_sentiment = (bullish_score - bearish_score) * impact_multiplier * source_weight
        
        # Normalize to [-1, 1] range
        if net_sentiment > 0:
            return min(1.0, net_sentiment / 5.0)
        elif net_sentiment < 0:
            return max(-1.0, net_sentiment / 5.0)
        else:
            return 0.0
    
    def calculate_technical_indicators(self, previous_candles: List[Dict], observation_candles: List[Dict]) -> Dict[str, float]:
        """Calculate technical indicators from price data"""
        # Combine all available price data
        all_candles = previous_candles + observation_candles
        
        if len(all_candles) < 3:
            return {"rsi": 50.0, "macd": 0.0, "momentum": 0.0, "volume_ratio": 1.0}
        
        closes = [float(candle['close']) for candle in all_candles]
        volumes = [float(candle['volume']) for candle in all_candles]
        
        # RSI calculation (simplified)
        rsi = self.calculate_rsi(closes)
        
        # MACD-like momentum indicator
        macd = self.calculate_momentum(closes)
        
        # Price momentum over available period
        momentum = (closes[-1] - closes[0]) / closes[0] * 100 if closes[0] != 0 else 0.0
        
        # Volume analysis
        volume_ratio = volumes[-1] / np.mean(volumes[:-1]) if len(volumes) > 1 and np.mean(volumes[:-1]) > 0 else 1.0
        
        return {
            "rsi": rsi,
            "macd": macd,
            "momentum": momentum,
            "volume_ratio": volume_ratio
        }
    
    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < 2:
            return 50.0
        
        # Use available data if less than period
        actual_period = min(period, len(prices) - 1)
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        if len(gains) == 0:
            return 50.0
        
        avg_gain = np.mean(gains[-actual_period:])
        avg_loss = np.mean(losses[-actual_period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_momentum(self, prices: List[float]) -> float:
        """Calculate price momentum indicator"""
        if len(prices) < 3:
            return 0.0
        
        # Simple momentum: recent price change vs earlier price change
        recent_change = prices[-1] - prices[-2] if len(prices) >= 2 else 0
        earlier_change = prices[-2] - prices[-3] if len(prices) >= 3 else 0
        
        if earlier_change == 0:
            return recent_change
        
        return (recent_change - earlier_change) / abs(earlier_change)
    
    def calculate_volatility_score(self, observation_candles: List[Dict]) -> float:
        """Calculate volatility score from observation candles"""
        if len(observation_candles) < 2:
            return 0.0
        
        # Calculate price ranges and volatility
        price_ranges = []
        for candle in observation_candles:
            high = float(candle['high'])
            low = float(candle['low'])
            close = float(candle['close'])
            price_range = (high - low) / close if close > 0 else 0
            price_ranges.append(price_range)
        
        return np.mean(price_ranges) * 100  # Convert to percentage
    
    def score_news_event(self, event: Dict) -> Dict[str, Any]:
        """Score a single news event for trading potential"""
        # Extract data
        title = event.get('title', '')
        source = event.get('source', '')
        previous_candles = event.get('previous_candles', [])
        observation_candles = event.get('observation_candles', [])
        
        # Calculate components
        sentiment = self.analyze_sentiment(title, source)
        technical = self.calculate_technical_indicators(previous_candles, observation_candles)
        volatility = self.calculate_volatility_score(observation_candles)
        
        # Get entry price (first observation candle close)
        entry_price = 0.0
        if observation_candles:
            entry_price = float(observation_candles[0].get('close', 0))
        
        # Combine signals for final score
        # Technical weight: 40%, Sentiment: 35%, Volatility: 25%
        technical_signal = (
            (technical['momentum'] * 0.3) +
            ((technical['rsi'] - 50) / 50 * 0.2) +  # Normalize RSI to [-1,1]
            (technical['macd'] * 0.3) +
            ((technical['volume_ratio'] - 1) * 0.2)  # Volume above average is positive
        )
        
        # Volatility contributes to trading opportunity (higher volatility = more opportunity)
        volatility_signal = min(volatility / 5.0, 1.0)  # Cap at 1.0
        
        # Combined signal
        combined_signal = (
            technical_signal * 0.4 +
            sentiment * 0.35 +
            volatility_signal * 0.25
        )
        
        # Determine trading direction
        if combined_signal > 0.1:
            decision = "LONG"
        elif combined_signal < -0.1:
            decision = "SHORT"
        else:
            # For neutral signals, use momentum as tiebreaker
            decision = "LONG" if technical['momentum'] >= 0 else "SHORT"
        
        return {
            'event_id': event.get('id'),
            'decision': decision,
            'confidence': abs(combined_signal),
            'sentiment': sentiment,
            'technical_signal': technical_signal,
            'volatility': volatility,
            'entry_price': entry_price,
            'combined_signal': combined_signal
        }
    
    def select_best_trades(self, scored_events: List[Dict], target_count: int = 50) -> List[Dict]:
        """Select the best trading opportunities"""
        # Sort by confidence (absolute signal strength) descending
        scored_events.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Take top target_count events
        selected = scored_events[:target_count]
        
        # Ensure we have a mix of LONG and SHORT positions for diversification
        long_trades = [trade for trade in selected if trade['decision'] == 'LONG']
        short_trades = [trade for trade in selected if trade['decision'] == 'SHORT']
        
        # If too skewed, balance the portfolio
        if len(long_trades) > target_count * 0.8:  # More than 80% long
            # Replace some long trades with best short trades
            excess_long = len(long_trades) - int(target_count * 0.7)
            remaining_short = [trade for trade in scored_events if trade['decision'] == 'SHORT' and trade not in selected]
            remaining_short.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Replace lowest confidence long trades with highest confidence short trades
            long_trades.sort(key=lambda x: x['confidence'])
            selected = long_trades[excess_long:] + short_trades + remaining_short[:excess_long]
        
        elif len(short_trades) > target_count * 0.8:  # More than 80% short
            # Replace some short trades with best long trades
            excess_short = len(short_trades) - int(target_count * 0.7)
            remaining_long = [trade for trade in scored_events if trade['decision'] == 'LONG' and trade not in selected]
            remaining_long.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Replace lowest confidence short trades with highest confidence long trades
            short_trades.sort(key=lambda x: x['confidence'])
            selected = short_trades[excess_short:] + long_trades + remaining_long[:excess_short]
        
        return selected[:target_count]

@app.route("/trading-bot", methods=["POST"])
def trading_bot():
    """
    Main trading bot endpoint that processes news events and returns trading decisions.
    
    Expected input: List of 1000 news events with price data
    Returns: List of 50 trading decisions (id + LONG/SHORT decision)
    """
    try:
        # Parse request data
        data = request.get_json(silent=True)
        if not data or not isinstance(data, list):
            return bad_request("Expected a JSON array of news events")
        
        if len(data) == 0:
            return bad_request("No news events provided")
        
        # Initialize trading bot
        bot = TradingBot()
        
        # Score all events
        scored_events = []
        for event in data:
            try:
                if not isinstance(event, dict) or 'id' not in event:
                    continue  # Skip invalid events
                
                score_result = bot.score_news_event(event)
                scored_events.append(score_result)
                
            except Exception as e:
                # Log error but continue processing other events
                print(f"Error processing event {event.get('id', 'unknown')}: {str(e)}")
                continue
        
        if len(scored_events) == 0:
            return bad_request("No valid events could be processed")
        
        # Select best 50 trading opportunities
        target_count = min(50, len(scored_events))
        selected_trades = bot.select_best_trades(scored_events, target_count)
        
        # Format response
        response = []
        for trade in selected_trades:
            response.append({
                "id": trade['event_id'],
                "decision": trade['decision']
            })
        
        # Ensure we have exactly 50 decisions (pad if necessary)
        while len(response) < 50 and len(scored_events) > len(response):
            # Add remaining events with random decisions if we don't have enough
            remaining_events = [e for e in scored_events if e['event_id'] not in [r['id'] for r in response]]
            if remaining_events:
                trade = remaining_events[0]
                response.append({
                    "id": trade['event_id'],
                    "decision": trade['decision']
                })
        
        return jsonify(response), 200
        
    except Exception as e:
        return bad_request(f"Trading bot error: {str(e)}", status_code=500)

# Error handlers
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

if __name__ == "__main__":
    # For local development only
    app.run(debug=True)