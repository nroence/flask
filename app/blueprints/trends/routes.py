# flask-analytics/app/blueprints/trends/routes.py

from flask import Blueprint, request, jsonify
from ...services.ensemble import run_ensemble

trends_bp = Blueprint("trends", __name__)


@trends_bp.route("/result", methods=["GET"])
def result():
    """
    GET /trends/result
      ?from=YYYY-MM-DD
      &to=YYYY-MM-DD
      &queue_id=<int>
      &model=arima|lstm|ensemble        (default="ensemble")
      &steps=<int>                       (default=14)
      &window_size=<int>                 (default=14)

    Returns JSON with:
    {
      "historical_mean": <float>,
      "arima":    { "dates": [...], "values": [...] }      (if model in {"arima","ensemble"}),
      "lstm":     { "dates": [...], "values": [...] }      (if model in {"lstm","ensemble"}),
      "ensemble": { "dates": [...], "values": [...] }      (if model == "ensemble")
    }
    """
    # 1) Read and type‐cast query parameters
    date_from   = request.args.get("from")
    date_to     = request.args.get("to")
    queue_id    = request.args.get("queue_id")
    model       = request.args.get("model", "ensemble").lower()
    try:
        steps = int(request.args.get("steps", 14))
    except ValueError:
        steps = 14
    try:
        window_size = int(request.args.get("window_size", 14))
    except ValueError:
        window_size = 14

    # 2) Call run_ensemble (no retraining on GET)
    try:
        payload = run_ensemble(
            date_from=date_from,
            date_to=date_to,
            queue_id=queue_id,
            model=model,
            steps=steps,
            window_size=window_size,
            retrain=False
        )
    except ValueError as e:
        # If build_daily_series raised ValueError (e.g. “No data in date range”)
        return jsonify({"error": str(e)}), 200

    # 3) Return the JSON‐friendly payload exactly as run_ensemble built it
    return jsonify(payload), 200


@trends_bp.route("/analyse", methods=["POST"])
def analyse():
    """
    POST /trends/analyse
    Body (JSON):
      {
        "from": "YYYY-MM-DD",
        "to": "YYYY-MM-DD",
        "queue_id": <int>,
        "model": "arima"|"lstm"|"ensemble",  (optional, default="ensemble")
        "steps": <int>,                     (optional, default=14)
        "window_size": <int>                (optional, default=14)
      }

    Retrains the requested model(s) and returns the same JSON structure.
    """
    data = request.get_json() or {}
    date_from   = data.get("from")
    date_to     = data.get("to")
    queue_id    = data.get("queue_id")
    model       = data.get("model", "ensemble").lower()
    try:
        steps = int(data.get("steps", 14))
    except (TypeError, ValueError):
        steps = 14
    try:
        window_size = int(data.get("window_size", 14))
    except (TypeError, ValueError):
        window_size = 14

    # 1) Call run_ensemble with retrain=True
    try:
        payload = run_ensemble(
            date_from=date_from,
            date_to=date_to,
            queue_id=queue_id,
            model=model,
            steps=steps,
            window_size=window_size,
            retrain=True
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 200

    # 2) Return the newly trained forecasts
    return jsonify(payload), 200
