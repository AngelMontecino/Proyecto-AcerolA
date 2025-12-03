import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Cargar el dataset (enlace directo al UCI)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00198/Faults.NNA"
# Nombres de columnas basados en la documentación del dataset
column_names = [
    "X_Minimum", "X_Maximum", "Y_Minimum", "Y_Maximum", "Pixels_Areas", "X_Perimeter",
    "Y_Perimeter", "Sum_of_Luminosity", "Minimum_of_Luminosity", "Maximum_of_Luminosity",
    "Length_of_Conveyer", "TypeOfSteel_A300", "TypeOfSteel_A400", "Steel_Plate_Thickness",
    "Edges_Index", "Empty_Index", "Square_Index", "Outside_X_Index", "Edges_X_Index",
    "Edges_Y_Index", "Outside_Global_Index", "LogOfAreas", "Log_X_Index", "Log_Y_Index",
    "Orientation_Index", "Luminosity_Index", "SigmoidOfAreas", 
    "Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"
]

data = pd.read_csv(url, sep='\t', header=None, names=column_names)

# 2. Preprocesamiento
# El dataset tiene 7 columnas binarias para el objetivo (One-Hot). Las convertimos a una sola columna 'Target'.
target_cols = ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]
data['Defect_Type'] = data[target_cols].idxmax(axis=1)

# Seleccionamos las variables que pusieron en su Wireframe (o sus equivalentes más cercanos)
# Wireframe: Grosor, Área, Luminosidad, Longitud X.
features = ['Steel_Plate_Thickness', 'Pixels_Areas', 'Sum_of_Luminosity', 'X_Maximum'] # Ajustar según necesidad
X = data[features]
y = data['Defect_Type']

# Escalar datos (OBLIGATORIO para SVM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Implementación SVM
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# probability=True es necesario para mostrar el porcentaje de confianza en la interfaz
svm_model = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
svm_model.fit(X_train, y_train)

# Evaluación rápida
y_pred = svm_model.predict(X_test)
print("Precisión del modelo preliminar:", accuracy_score(y_test, y_pred))

# 4. Guardar modelo y escalador para usar en Flask
joblib.dump(svm_model, 'modelo_svm.pkl')
joblib.dump(scaler, 'escalador.pkl')
print("Archivos guardados correctamente.")