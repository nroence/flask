#flask-analytics/app/utils/serializer.py
import json
from datetime import date, datetime

class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (date, datetime)):
            return o.isoformat()
        return super().default(o)

def dumps(obj):
    return json.dumps(obj, cls=EnhancedJSONEncoder)
