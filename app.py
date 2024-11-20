from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)  # Разрешаем CORS для всех маршрутов

# Загрузка моделей
rf_model = joblib.load('rf_model.joblib')
gb_model = joblib.load('gb_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = pd.DataFrame([data])  # Конвертируем JSON в DataFrame
    
    # Предсказание модели
    rf_pred = rf_model.predict(input_data)[0]
    gb_pred = gb_model.predict(input_data)[0]
    ensemble_pred = (rf_pred + gb_pred) / 2

    return jsonify({'prediction': ensemble_pred})

if __name__ == '__main__':
    app.run(debug=True)
