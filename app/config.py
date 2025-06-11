# flask-analytics/app/config.py
import os

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret")

    # Enable SQLAlchemy connection
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", 
        "mysql+pymysql://u782952718_lance:Vanrodolf123.@217.21.80.1:3306/u782952718_lance"
    )

    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Directory for persisted models
    MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(os.getcwd(), "models"))

    # CORS origins, timeouts, etc.
    JSON_SORT_KEYS = False
