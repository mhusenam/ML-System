import mlflow
import pandas as pd
import json
from mlflow.models import validate_serving_input, convert_input_example_to_serving_input

# --- SETUP ---
# GANTI dengan Run ID kamu yang ada di MLflow UI (contoh: stylish-sheep)
run_id = "d306be46263a4f25962582623d910ba2" 
model_uri = f'runs:/{run_id}/model'

# --- 1. DATA INPUT (SIAPKAN PERTANYAAN) ---
data_sample = {
    "columns": ["Age", "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour", 
                "pc1_1", "pc1_2", "pc1_3", "pc1_4", "pc1_5", "pc2_1", "pc2_2"],
    "data": [[0.7142857142857142, 1, 1, 3, -0.43815, 0.17113, 0.07736, -0.04019, 0.04959, -0.14482, -0.06066]]
}

# --- 2. FORMATTING (TRANSLASI KE BAHASA SERVER) ---
# Mengubah data Python ke format JSON standar MLflow Serving
serving_payload = json.loads(convert_input_example_to_serving_input(data_sample))
input_data = {"dataframe_split": serving_payload["inputs"]}

# --- 3. VALIDASI (CEK APAKAH FORMATNYA COCOK) ---
print("Sedang memvalidasi input...")
validate_serving_input(model_uri, input_data)

# --- 4. PREDIKSI (SI MODEL MENJAWAB) ---
print("Memuat model dan melakukan prediksi...")
model = mlflow.pyfunc.load_model(model_uri)

# Ubah input kembali ke DataFrame agar bisa diprediksi oleh model
df_test = pd.DataFrame(input_data["dataframe_split"]["data"], 
                       columns=input_data["dataframe_split"]["columns"])

hasil_prediksi = model.predict(df_test)

print("-" * 30)
print(f"HASIL PREDIKSI: {hasil_prediksi}")
print("-" * 30)