from flask import Flask
from .blueprints.trends.routes import trends_bp
# import and register any other blueprints or extensions here

def create_app(config_object=None):
    app = Flask(__name__, instance_relative_config=False)

    # load config (optional)
    if config_object:
        app.config.from_object(config_object)
    else:
        # default config
        app.config.from_mapping({
            "DEBUG": True,
            # add other defaults here
        })

    # register blueprints
    app.register_blueprint(trends_bp, url_prefix="/trends")
    # app.register_blueprint(other_bp, url_prefix="/...")

    # initialize any extensions here (db, migrate, etc.)

    return app
