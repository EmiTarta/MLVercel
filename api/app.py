from flask import Flask, request, jsonify, render_template
import pickle
import datetime
import pandas as pd
from sqlalchemy import create_engine
import sqlite3
import matplotlib.pyplot as plt
import io
from flask import Response

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["DEBUG"] = False 

# Cargar el modelo entrenado
with open("../titanic_model.pkl", "rb") as f:
    model = pickle.load(f)

# Inicializar la base de datos
def init_db():
    # Conectar a la base de datos y crear la tabla 'predictions' si no existe
    # Completa aquí: conexión SQLite y creación de tabla con campos (inputs, prediction, timestamp)
    connection = sqlite3.connect("../predictions.db")  
    crsr = connection.cursor()
    query  = '''
        CREATE TABLE IF NOT EXISTS predictions (
            Pclass INT,
            Sex INT,
            Age FLOAT,
            prediction INT,
            timestamp TIMESTAMP
    );
'''
    crsr.execute(query)

init_db()

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint que recibe datos en formato JSON, realiza una predicción y guarda los datos en la base de datos.
    """
    try:
        data = request.json
        # 1. Extraer los datos de entrada del JSON recibido
        Pclass = data["Pclass"]
        Sex= data["Sex"]
        Age = data["Age"]
        input_data = [[Pclass, Sex, Age]]
        # 2. Realizar predicción con el modelo
        prediction = model.predict(input_data)[0]

        # 3. Guardar en la base de datos
        timestamp = datetime.datetime.now().isoformat()
        connection = sqlite3.connect("../predictions.db")
        crsr = connection.cursor()
        # Completa aquí: inserta los datos (inputs, predicción, timestamp) en la base de datos
        query = '''
                INSERT INTO predictions (Pclass, Sex, Age, prediction, timestamp)
                VALUES (?, ?, ?, ?, ?);
                '''
        crsr.execute(query, (Pclass, Sex, Age, int(prediction), timestamp))
        connection.commit()

        return jsonify({"prediction": int(prediction), "timestamp": timestamp})
    except Exception as e:
        return jsonify({"error": str(e)})
    
@app.route('/records', methods=['GET'])
def records():
    """
    Endpoint que devuelve todos los registros guardados en la base de datos.
    """
    try:
        # Conectar a la base de datos y recuperar los registros
        # Completa aquí: conexión SQLite y lectura de registros
        connection = sqlite3.connect("../predictions.db")
        crsr = connection.cursor()
        records = []  # Sustituir por los datos recuperados de la base de datos
        query = '''SELECT * FROM "predictions"'''
        crsr.execute(query)         # Obtengo el cursor, no los datos
        results = crsr.fetchall()       # Para obtener los datos
        for row in results:
            record = {
                        "Pclass": row[0],
                        "Sex": row[1],
                        "Age": row[2],
                        "prediction": row[3],
                        "timestamp": row[4]}
            records.append(record)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/graficopred', methods=['GET'])
def grafica():
    """
    Endpoint que que genera una grafica en matplotlib y la devuelve.
    """
    try:
        connection = sqlite3.connect("../predictions.db")
        crsr = connection.cursor()
        query = ''' SELECT prediction FROM predictions'''
        crsr.execute(query)
        data = crsr.fetchall()
        # Cuento valores de predicciones
        predictions = [pred[0] for pred in data] 
        unique_values = list(set(predictions))  # Obtener los valores únicos (0 y 1)
        frequencies = [predictions.count(value) for value in unique_values]  # Contar cada valor
        # Crear el gráfico
        plt.bar(unique_values, frequencies)
        plt.xlabel("Valores")
        plt.ylabel("Frecuencia")
        plt.title("Predicciones 'survived'")
        # Guardar el gráfico en un archivo temporal en memoria
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()
        # Devolver el gráfico como respuesta HTTP
        return Response(img, mimetype='image/png')

    except Exception as e:
        return jsonify({"error": str(e)})
    
@app.route('/')
def home():
    return render_template('./index.html')

@app.route('/grafpred', methods=['POST'])
def graf_predictions():
    # Obtener datos del formulario
    # features = [float(request.form[f'feature{i}']) for i in range(1, 6)]
    features = []
    for i in range(1, 4):
        value = request.form.get(f'feature{i}')
        if value is None or value.strip() == "":
            return f"Error: Missing input for feature{i}", 400
        features.append(float(value))
    
    # Realizar predicción
    prediction = model.predict([features])[0]
    
    # Mapear resultado a clase (opcional)
    classes = [0, 1]
    predicted_class = classes[prediction]

    return render_template('./result.html', prediction=predicted_class)

if __name__ == "__main__":
    app.run()