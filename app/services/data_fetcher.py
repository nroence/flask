# flask-analytics/app/services/data_fetcher.py

from datetime import datetime, timedelta
import mysql.connector
import os

def get_db_connection():
    return mysql.connector.connect(
        host               = os.getenv("DB_HOST", "127.0.0.1"),
        port               = int(os.getenv("DB_PORT", 3306)),
        user               = os.getenv("DB_USER"),
        password           = os.getenv("DB_PASSWORD"),
        database           = os.getenv("DB_NAME"),
        charset            = "utf8mb4",
        connection_timeout = 5
    )

def fetch_daily_visit_counts(date_from=None, date_to=None, queue_id: int | None = None):
    """
    Query the `tokens` table, grouped by DATE(created_at), optionally filtered by queue_id.
    Returns a list of dicts: [{"date": "YYYY-MM-DD", "count": 42}, ...]
    
    - If `queue_id` is provided, only tokens with that queue_id are counted.
    - If no date_from/to are given, defaults to the last 30 days.
    """
    conn = get_db_connection()
    cur  = conn.cursor(dictionary=True)

    today = datetime.today().date()
    if date_to is None:
        date_to = today.strftime("%Y-%m-%d")
    if date_from is None:
        date_from = (today - timedelta(days=30)).strftime("%Y-%m-%d")

    # Base SQL: count tokens by date(created_at)
    sql = """
      SELECT
        DATE(created_at) AS date,
        COUNT(*)           AS count
      FROM tokens
      WHERE DATE(created_at) BETWEEN %s AND %s
    """
    params = [date_from, date_to]

    if queue_id is not None:
        sql += " AND queue_id = %s"
        params.append(queue_id)

    sql += " GROUP BY DATE(created_at) ORDER BY DATE(created_at)"

    cur.execute(sql, params)
    rows = cur.fetchall()

    cur.close()
    conn.close()
    return rows
