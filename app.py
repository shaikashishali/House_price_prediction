
from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# 1. Load the saved model and other details
with open('model.pkl', 'rb') as file:
    model_data = pickle.load(file)

# 2. Get information from the pickle file
reg = model_data['model']
city_mapping = model_data['city_mapping']
zip_mapping = model_data['zip_mapping']
columns = model_data['columns']

accuracy = model_data['accuracy']
conf_matrix = model_data['confusion_matrix']
report = model_data['classification_report']

# 3. Input features required by the model
numeric_features = [
    'bedrooms',
    'bathrooms',
    'sqft_living',
    'sqft_lot',
    'floors',
    'waterfront',
    'view',
    'condition',
    'sqft_above',
    'sqft_basement',
    'yr_built',
    'yr_renovated'
]

# 4. Home page
@app.route('/', methods=['GET', 'POST'])
def home():

    result = None
    error = None

    if request.method == 'POST':

        try:
            # Get numerical values from the form
            input_data = {
                feature: float(request.form[feature])
                for feature in numeric_features
            }

            # Get selected city and ZIP code
            selected_city = request.form['city']
            selected_zip = request.form['statezip']

            # Convert city and ZIP using saved mappings
            input_data['city'] = city_mapping[selected_city]
            input_data['statezip'] = zip_mapping[selected_zip]

            # Country value used during training
            input_data['country'] = 0

            # Convert input into a DataFrame
            input_df = pd.DataFrame([input_data])

            # Keep the same column order used during training
            input_df = input_df[columns]

            # Predict the house price category
            prediction = reg.predict(input_df)[0]

            result = str(prediction)

        except ValueError:
            error = "Please enter valid numerical values."

        except KeyError:
            error = "Please select a valid city and ZIP code."

        except Exception as e:
            error = "Something went wrong: " + str(e)

    return render_template(
        'index.html',
        result=result,
        error=error,
        numeric_features=numeric_features,
        cities=list(city_mapping.keys()),
        zipcodes=list(zip_mapping.keys()),
        accuracy=round(accuracy * 100, 2),
        conf_matrix=conf_matrix.tolist(),
        report=report
    )


# 5. Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
