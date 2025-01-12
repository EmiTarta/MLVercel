from flask import Flask, request, jsonify, render_template, Response
import pickle
import datetime
import pandas as pd
import sqlite3
import io
import os
import matplotlib
import matplotlib.pyplot as plt
print("Archivos disponibles en el directorio actual:")
print(os.listdir("."))
# Desactiva la caché de matplotlib configurando un directorio temporal
os.environ["MPLCONFIGDIR"] = "/tmp"

# Configura el backend "Agg" para evitar dependencias gráficas
matplotlib.use("Agg")

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["DEBUG"] = False

# Obtén la ruta del directorio actual
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construye la ruta al archivo del modelo
model_path = os.path.join(current_dir, "titanic_model.pkl")

# Cargar el modelo entrenado
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Inicializar la base de datos
def init_db():
    """
    Crea la base de datos y la tabla 'predictions' si no existe.
    """
    connection = sqlite3.connect("predictions.db")

    crsr = connection.cursor()
    query = '''
        CREATE TABLE IF NOT EXISTS predictions (
            Pclass INT,
            Sex INT,
            Age FLOAT,
            prediction INT,
            timestamp TIMESTAMP
        );
    '''
    crsr.execute(query)
    connection.commit()
    connection.close()

# Inicializa la base de datos al iniciar la app
init_db()

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint que recibe datos en formato JSON, realiza una predicción y guarda los datos en la base de datos.
    """
    try:
        data = request.json
        # Extraer los datos de entrada del JSON recibido
        Pclass = data["Pclass"]
        Sex = data["Sex"]
        Age = data["Age"]
        input_data = [[Pclass, Sex, Age]]

        # Realizar predicción con el modelo
        prediction = model.predict(input_data)[0]

        # Guardar en la base de datos
        timestamp = datetime.datetime.now().isoformat()
        connection = sqlite3.connect("predictions.db")
        crsr = connection.cursor()
        query = '''
            INSERT INTO predictions (Pclass, Sex, Age, prediction, timestamp)
            VALUES (?, ?, ?, ?, ?);
        '''
        crsr.execute(query, (Pclass, Sex, Age, int(prediction), timestamp))
        connection.commit()
        connection.close()

        return jsonify({"prediction": int(prediction), "timestamp": timestamp})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/records', methods=['GET'])
def records():
    """
    Endpoint que devuelve todos los registros guardados en la base de datos.
    """
    try:
        connection = sqlite3.connect("predictions.db")
        crsr = connection.cursor()
        query = 'SELECT * FROM predictions'
        crsr.execute(query)
        results = crsr.fetchall()
        connection.close()

        records = [
            {"Pclass": row[0], "Sex": row[1], "Age": row[2], "prediction": row[3], "timestamp": row[4]}
            for row in results
        ]

        return jsonify(records)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/graficopred', methods=['GET'])
def grafica():
    """
    Endpoint que genera un gráfico en matplotlib y lo devuelve.
    """
    try:
        connection = sqlite3.connect("predictions.db")
        crsr = connection.cursor()
        query = 'SELECT prediction FROM predictions'
        crsr.execute(query)
        data = crsr.fetchall()
        connection.close()

        predictions = [pred[0] for pred in data]
        unique_values = list(set(predictions))
        frequencies = [predictions.count(value) for value in unique_values]

        # Crear el gráfico
        plt.bar(unique_values, frequencies)
        plt.xlabel("Valores")
        plt.ylabel("Frecuencia")
        plt.title("Predicciones 'survived'")
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()

        return Response(img, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/grafpred', methods=['POST'])
def graf_predictions():
    """
    Endpoint que procesa un formulario para predecir.
    """
    try:
        features = []
        for i in range(1, 4):
            value = request.form.get(f'feature{i}')
            if not value or value.strip() == "":
                return f"Error: Missing input for feature{i}", 400
            features.append(float(value))

        # Realizar predicción
        prediction = model.predict([features])[0]
        classes = [0, 1]
        predicted_class = classes[prediction]

        return render_template('result.html', prediction=predicted_class)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run()
