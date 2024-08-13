from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Load the saved model
with open('knn_covid_model.pkl2', 'rb') as file:
    knn_model = pickle.load(file)

# Define the home route
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        options = {"No": 0, "Yes": 1}
        # Collect data from the form
        input_data = np.array([
            options[request.form['Breathing_Problem']],
            options[request.form['Fever']],
            options[request.form['Dry_Cough']],
            options[request.form['Sore_throat']],
            options[request.form['Running_Nose']],
            options[request.form['Asthma']],
            options[request.form['Chronic_Lung_Disease']],
            options[request.form['Headache']],
            options[request.form['Heart_Disease']],
            options[request.form['Diabetes']],
            options[request.form['Hyper_Tension']],
            options[request.form['Fatigue']],
            options[request.form['Gastrointestinal']],
            options[request.form['Abroad_travel']],
            options[request.form['Contact_with_COVID_Patient']],
            options[request.form['Attended_Large_Gathering']],
            options[request.form['Visited_Public_Exposed_Places']],
            options[request.form['Family_working_in_Public_Exposed_Places']]
        ]).reshape(1, -1)
        
        # Make the prediction
        prediction = knn_model.predict(input_data)[0]
    
    return render_template("index.html", prediction=prediction)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
