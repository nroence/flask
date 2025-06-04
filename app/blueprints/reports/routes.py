#flask-analytics/app/blueprints/reports/routes.py
from flask import Blueprint, request, jsonify, send_file
from ...services.report_builder import build_report

reports_bp = Blueprint("reports", __name__)

@reports_bp.route("/", methods=["GET"])
def index():
    """7.2 View Report (with optional from/to query params)"""
    date_from = request.args.get("from")
    date_to   = request.args.get("to")
    report = build_report(date_from, date_to)
    return jsonify(report)

@reports_bp.route("/generate", methods=["POST"])
def generate():
    """7.1 Perform Report generation"""
    data = request.json or {}
    date_from = data.get("from")
    date_to   = data.get("to")
    build_report(date_from, date_to, force_recalc=True)
    return jsonify({"status": "queued"}), 202

@reports_bp.route("/verify", methods=["GET"])
def verify():
    """7.1.1 Verify data sanity"""
    # placeholder: replace with real checks
    missing = 0
    return jsonify({"ok": missing == 0, "missing": missing})

@reports_bp.route("/export", methods=["GET"])
def export():
    """7.3 Export report as PDF or Excel"""
    fmt = request.args.get("fmt", "pdf").lower()
    date_from = request.args.get("from")
    date_to   = request.args.get("to")
    if fmt == "xlsx":
        path = build_report(date_from, date_to, export="xlsx")
    else:
        path = build_report(date_from, date_to, export="pdf")
    return send_file(path, as_attachment=True)
