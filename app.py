from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict_churn():
    try:
        # Get data from the request JSON
        data = request.json
        preprocessed_data = preprocess_data(data)
       
        model = joblib.load('model.pkl', 'rb')
        # print(model.feature_names_in_)
        
        # Predict churn using the model
        prediction = model.predict(preprocessed_data)
        print("prediction",prediction)
        prediction = True if prediction[0] == 1 else False
        print("prediction",prediction)
        # Return the prediction as a JSON response
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

def preprocess_data(input_data):
    """
    Preprocess the input data before making a prediction.
    
    Parameters:
    input_data (dict): Dictionary containing input data fields.
    
    Returns:
    preprocessed_data (pd.DataFrame): Preprocessed data ready for prediction.
    """
    # Convert the input data into a DataFrame
    input_df = pd.DataFrame([input_data])

    # Perform data preprocessing steps
    preprocessed_data = input_df.copy()
    preprocessed_data.drop(columns=['CustomerID','Name','Location'],axis=1,inplace=True)
    preprocessed_data['Gender'] = (preprocessed_data['Gender'] == 'Male').astype(int)

    return preprocessed_data

if __name__ == '__main__':
    app.run(debug=True)