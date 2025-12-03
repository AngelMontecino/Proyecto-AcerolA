from flask import Flask, request, render_template
import joblib 
import numpy as np

app = Flask(__name__)

# Cargar modelo y escalador
try:
    model = joblib.load('modelo_svm.pkl') 
    scaler = joblib.load('escalador.pkl')
    print("✅ Archivos cargados correctamente (Modo 4 variables)")
except Exception as e:
    print(f"❌ Error fatal cargando modelo: {e}")
    model = None
    scaler = None

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    
    if request.method == 'POST':
        try:
            # 1. Obtiene datos del formulario
            grosor = float(request.form['grosor'])
            area = float(request.form['area'])
            luminosidad = float(request.form['luminosidad'])
            longitud = float(request.form['longitud'])
            
            # 2. CREAR EL ARREGLO SIMPLE (Solo las 4 variables)
            input_data = np.array([[grosor, area, luminosidad, longitud]])

            # 3. Preprocesar
            input_scaled = scaler.transform(input_data)
            
            # 4. Predicción
            pred_class = model.predict(input_scaled)[0]
            
            # Obtener probabilidad (confianza)
            # Si esto falla, lo envolveremos en un try interno.
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(input_scaled)
                confidence = round(np.max(probs) * 100, 2)
            else:
                confidence = "N/A (SVM Lineal)"
            
            prediction = pred_class
            
        except Exception as e:
            prediction = f"Error técnico: {str(e)}"
            confidence = 0

    return render_template('index.html', prediction=prediction, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)