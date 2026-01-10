import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# 1. SETUP SEED (Agar hasil konsisten)
np.random.seed(42)
tf.random.set_seed(42)

print("🚀 MEMULAI TRAINING: STACKED LSTM")

# ==========================================
# 2. LOAD DATA (KHUSUS EXCEL .XLSX)
# ==========================================
try:
    nama_file = "penelitian_bersih.xlsx"
    
    # Baca Excel (Engine openpyxl wajib terinstall)
    df = pd.read_excel(nama_file, engine='openpyxl')
    print(f"✅ Data '{nama_file}' berhasil dimuat!")

except Exception as e:
    print(f"❌ Error saat membaca file: {e}")
    print("   Pastikan library 'openpyxl' sudah diinstall: pip install openpyxl")
    exit()

# ==========================================
# 3. PREPROCESSING
# ==========================================
# Ganti V/X jadi angka
df = df.replace({'X': 0, 'V': 1, 'x': 0, 'v': 1})

# Hapus kolom yang tidak dipakai
dataset_clean = df.drop(columns=['namakecamatan', 'namakelurahan', 'Tahun', 'lat', 'lon'], errors='ignore').dropna()

X = dataset_clean.drop(['risiko_stunting'], axis=1)
y = dataset_clean['risiko_stunting']

# --- Logic Pembersihan Lanjutan ---
print("🧹 Membersihkan data non-numeric...")
X_temp = X.copy()
y_temp = y.copy()

# Paksa semua kolom jadi angka, error jadi NaN
for col in X_temp.select_dtypes(include=['object']).columns:
    X_temp[col] = pd.to_numeric(X_temp[col], errors='coerce')

# Hapus baris yang mengandung NaN setelah dipaksa jadi angka
combined_df = pd.concat([X_temp, y_temp.rename('target')], axis=1)
cleaned_combined_df = combined_df.dropna()

# Pisahkan lagi X dan y
X = cleaned_combined_df.drop('target', axis=1)
y = cleaned_combined_df['target']

print(f"ℹ️ Data akhir setelah cleaning: {X.shape[0]} baris")

# --- Scaling & Reshape ---
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Reshape ke 3D [Samples, Timesteps=1, Features]
X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Split Data (Random State 42 agar sama dengan BiLSTM)
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42, stratify=y)

# Simpan Scaler & Data Test
# (Scaler ditimpa tidak masalah karena logikanya sama dengan BiLSTM)
print("💾 Menyimpan scaler & data test...")
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('data_test.pkl', 'wb') as f:
    pickle.dump((X_test, y_test), f)

# ==========================================
# 4. BUILD MODEL STACKED LSTM
# ==========================================
model = Sequential(name="Stacked_LSTM")
model.add(Input(shape=(1, X.shape[1])))

# Layer 1 (return_sequences=True wajib untuk Stacked)
model.add(LSTM(64, return_sequences=True)) 
model.add(Dropout(0.3))

# Layer 2
model.add(LSTM(32))
model.add(Dropout(0.2))

# Output
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# ==========================================
# 5. TRAINING
# ==========================================
print("⏳ Sedang melatih model Stacked LSTM...")
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train, 
    epochs=100, 
    batch_size=32, 
    validation_split=0.2, 
    callbacks=[early_stop], 
    verbose=1
)

# ==========================================
# 6. SIMPAN HASIL
# ==========================================
model.save("model_stacked.h5")

with open('history_stacked.pkl', 'wb') as f:
    pickle.dump(history.history, f)

print("\n✅ SELESAI! Model disimpan sebagai 'model_stacked.h5'")

# ==========================================
# 7. EVALUASI MODEL (BAGIAN BARU)
# ==========================================
print("\n" + "="*40)
print("📊 HASIL EVALUASI STACKED LSTM (DATA TEST)")
print("="*40)

# Prediksi ke data test
y_pred_probs = model.predict(X_test, verbose=0)
# Ubah probabilitas jadi 0 atau 1 (Threshold 0.5)
y_pred = (y_pred_probs > 0.5).astype(int)

# Hitung Metrik
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print(f"🎯 Accuracy  : {acc:.4f}  (Ketepatan Umum)")
print(f"🎯 Precision : {prec:.4f}  (Ketepatan Prediksi 'Berisiko')")
print(f"🎯 Recall    : {rec:.4f}  (Sensitivitas Deteksi 'Berisiko')")
print(f"🎯 F1-Score  : {f1:.4f}  (Keseimbangan Prec & Rec)")

print("\n📋 DETAIL (Classification Report):")
print(classification_report(y_test, y_pred, target_names=['Aman (0)', 'Berisiko (1)'], zero_division=0))

print("\n🧩 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"True Negative (Aman terdeteksi Aman)     : {cm[0][0]}")
print(f"False Positive (Aman dibilang Risiko)    : {cm[0][1]}")
print(f"False Negative (Risiko dibilang Aman)    : {cm[1][0]}")
print(f"True Positive (Risiko terdeteksi Risiko) : {cm[1][1]}")
print("="*40)