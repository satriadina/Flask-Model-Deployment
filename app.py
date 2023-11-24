from flask import Flask, render_template, request
import pickle
import pandas as pd

# Flask web application
app = Flask(__name__)

# Load data from the pickle file
model_file = 'iris-svm.pickle'
with open(model_file, 'rb') as file:
    data = pickle.load(file)

model = data['model']
scaler = data['scaler']
species_map = data['species_map']

# route for the root URL ("/")
@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_species_name = None

    if request.method == 'POST':
        sepal_length = request.form.get('sepal_length')
        sepal_width = request.form.get('sepal_width')
        petal_length = request.form.get('petal_length')
        petal_width = request.form.get('petal_width')
        
        # input data to list
        input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

        # use dataframe to add feature names/column
        X = pd.DataFrame(
            input_data, 
            columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

        # transform using fitted StandardScaler
        x_scaled = scaler.transform(X)

        # use model to predict 
        y_preds = model.predict(x_scaled)
        y_pred = y_preds[0]

        # map predicted species id to species name
        predicted_species_name = species_map[y_pred]
    
    return render_template(
        'index.html', 
        PRED_RESULT=predicted_species_name
        )

if __name__ == '__main__':
    # Run the application on a local development server
    app.run(debug=True)