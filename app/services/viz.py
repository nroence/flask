#flask-analytics/app/services/arima_model.py
import matplotlib.pyplot as plt
import io
import base64

def plot_line(data):
    """
    Return a base64-encoded PNG line chart from a list of numbers.
    """
    fig, ax = plt.subplots()
    ax.plot(data)
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('ascii')
    plt.close(fig)
    return img_b64
