from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load('model/wine_cultivar_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        data = [
            float(request.form['alcohol']),
            float(request.form['malic_acid']),
            float(request.form['alcalinity_of_ash']),
            float(request.form['magnesium']),
            float(request.form['flavanoids']),
            float(request.form['proline'])
        ]

        result = model.predict([data])[0]
        prediction = f"Cultivar {result + 1}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
