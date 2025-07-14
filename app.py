from flask import Flask,render_template,request
import joblib 
import gzip
import pandas as pd

app=Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect input data from form
        data = {
            "longitude": float(request.form["longitude"]),
            "latitude": float(request.form["latitude"]),
            "housing_median_age": float(request.form["housing_median_age"]),
            "total_rooms": float(request.form["total_rooms"]),
            "total_bedrooms": float(request.form["total_bedrooms"]),
            "population": float(request.form["population"]),
            "households": float(request.form["households"]),
            "median_income":float(request.form["median_income"]),
            "ocean_proximity": request.form["ocean_proximity"]
        }

        with gzip.open('model_compressed.pkl.gz', 'rb') as f:
             model = joblib.load(f)

        input_df = pd.DataFrame([data])
        prediction = round(model.predict(input_df)[0], 2)

        return render_template("result.html", prediction=prediction, data=data)

    except Exception as e:
        return render_template("result.html", prediction=f"Error: {str(e)}")


if __name__=="__main__":
    app.run(debug=True)