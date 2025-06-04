from flask import Flask
from .config import Config

def create_app():
    app = Flask(__name__, instance_relative_config=False)
    app.config.from_object(Config)

    # Register Blueprints
    from .blueprints.reports.routes import reports_bp
    app.register_blueprint(reports_bp, url_prefix="/report")

    from .blueprints.trends.routes import trends_bp
    app.register_blueprint(trends_bp, url_prefix="/trend")

    return app
