from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Cargar modelo y escalador
model = joblib.load('modelo_svm.pkl')
scaler = joblib.load('escalador.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    
    if request.method == 'POST':
        try:
            # Obtener datos del formulario (coincide con el wireframe)
            grosor = float(request.form['grosor'])
            area = float(request.form['area'])
            luminosidad = float(request.form['luminosidad'])
            longitud = float(request.form['longitud'])
            
            # Preprocesar igual que en el entrenamiento
            input_data = np.array([[grosor, area, luminosidad, longitud]])
            input_scaled = scaler.transform(input_data)
            
            # Predicción
            pred_class = model.predict(input_scaled)[0]
            probs = model.predict_proba(input_scaled)
            confidence = round(np.max(probs) * 100, 2)
            
            prediction = pred_class
            
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)