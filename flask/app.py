import joblib
from flask import Flask, request, render_template
import numpy as np

app = Flask(__name__)

# تحميل النموذج
model = joblib.load("maram.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            precipitation = float(request.form["precipitation"])
            temp_max = float(request.form["temp_max"])
            temp_min = float(request.form["temp_min"])
            wind = float(request.form["wind"])

            user_prediction = np.array([[precipitation, temp_max, temp_min, wind]])
            prediction = model.predict(user_prediction)[0]

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
