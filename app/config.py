#flask-analytics/app/config.py
import os

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret")
    # If you connect to a database, you could add:
    # SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL")

    # Directory for persisted models
    MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(os.getcwd(), "models"))

    # CORS origins, timeouts, etc.
    JSON_SORT_KEYS = False
