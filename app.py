import pickle
from pathlib import Path

from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

model_path = Path(__file__).with_name("model.pkl")
with model_path.open("rb") as file:
    model = pickle.load(file)

FEATURE_LABELS = {
    "height_dm": ("Height", "decimeters"),
    "weight_hg": ("Weight", "hectograms"),
    "base_experience": ("Base experience", "points"),
    "hp": ("HP", "hit points"),
    "attack": ("Attack", "stat points"),
    "defense": ("Defense", "stat points"),
}
FEATURE_NAMES = list(model.feature_names_in_)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            features = [float(request.form[name]) for name in FEATURE_NAMES]
            feature_frame = pd.DataFrame([features], columns=FEATURE_NAMES)
            prediction = str(model.predict(feature_frame)[0]).title()
        except (ValueError, KeyError) as exc:
            error = "Enter a valid number for each stat before predicting."

    fields = [
        {
            "name": name,
            "label": FEATURE_LABELS[name][0],
            "unit": FEATURE_LABELS[name][1],
            "value": request.form.get(name, ""),
        }
        for name in FEATURE_NAMES
    ]

    return render_template(
        "index.html", prediction=prediction, error=error, fields=fields
    )

if __name__ == "__main__":
    app.run(debug=False)