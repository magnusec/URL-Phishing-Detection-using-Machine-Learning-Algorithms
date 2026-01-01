from flask import Flask, render_template, request, send_file
import joblib
import pandas as pd
import os
from datetime import datetime

app = Flask(__name__)

# =========================
# Load model bundle
# =========================
bundle = joblib.load("../models/dataset3_features.pkl")
model = bundle["model"]
FEATURES = bundle["features"]

LOG_FILE = "logs.csv"

# =========================
# Feature extractor (DATASET 3)
# =========================
def extract_features(url):
    features = {
        "url_length": len(url),
        "n_dots": url.count("."),
        "n_hyphens": url.count("-"),
        "n_underline": url.count("_"),
        "n_slash": url.count("/"),
        "n_questionmark": url.count("?"),
        "n_equal": url.count("="),
        "n_at": url.count("@"),
        "n_and": url.count("&"),
        "n_exclamation": url.count("!"),
        "n_space": url.count(" "),
        "n_tilde": url.count("~"),
        "n_comma": url.count(","),
        "n_plus": url.count("+"),
        "n_asterisk": url.count("*"),
        "n_hashtag": url.count("#"),
        "n_dollar": url.count("$"),
        "n_percent": url.count("%"),
        "n_redirection": url.count("://")
    }

    df = pd.DataFrame([features])
    df = df.reindex(columns=FEATURES, fill_value=0)
    return df


# =========================
# Home Route
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    url_value = None

    if request.method == "POST":
        url_value = request.form["url"]

        df = extract_features(url_value)
        result = model.predict(df)[0]
        prediction = "PHISHING 🚨" if result == 1 else "LEGITIMATE ✅"

        # -------- LOG TO CSV --------
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "url": url_value,
            "prediction": prediction
        }

        if not os.path.exists(LOG_FILE):
            pd.DataFrame([log_entry]).to_csv(LOG_FILE, index=False)
        else:
            pd.DataFrame([log_entry]).to_csv(
                LOG_FILE, mode="a", header=False, index=False
            )

    return render_template("index.html", prediction=prediction, url_value=url_value)


# =========================
# Export Logs
# =========================
@app.route("/export-logs")
def export_logs():
    if os.path.exists(LOG_FILE):
        return send_file(LOG_FILE, as_attachment=True)
    return "No logs available"


if __name__ == "__main__":
    app.run(debug=True, port=5001)
