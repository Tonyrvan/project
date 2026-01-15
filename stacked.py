import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# 1. LOAD DATA
print("📂 Loading data...")
df = pd.read_excel('penelitian_bersih.xlsx')
df = df.replace({'X': 0, 'V': 1, 'x': 0, 'v': 1, 'VV': 1})

# Hapus kolom metadata (sesuai diskusi sebelumnya)
df_clean = df.drop(columns=['namakecamatan', 'namakelurahan', 'Tahun', 'lat', 'lon'], errors='ignore').dropna()
df_clean = df_clean.apply(pd.to_numeric, errors='coerce').dropna()

# 2. PREPROCESSING
print("🧹 Preprocessing...")
Q1 = df_clean.quantile(0.25)
Q3 = df_clean.quantile(0.75)
IQR = Q3 - Q1
df_final = df_clean[~((df_clean < (Q1 - 1.5 * IQR)) | (df_clean > (Q3 + 1.5 * IQR))).any(axis=1)]

X = df_final.drop(['risiko_stunting'], axis=1)
y = df_final['risiko_stunting']

# Balancing data
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Scaling & Reshaping
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_res)
X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Split data (Wajib stratify agar seimbang)
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_res, test_size=0.2, random_state=42, stratify=y_res)

# 3. MODELING (Stacked LSTM 3 Layer)

model = Sequential(name="Stacked_LSTM")
model.add(Input(shape=(1, X.shape[1])))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0005), metrics=['accuracy'])

# 4. TRAINING
print("⏳ Training Stacked LSTM...")
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

history = model.fit(
    X_train, y_train, 
    epochs=100, 
    batch_size=256, 
    validation_split=0.2, 
    callbacks=[early_stop], 
    verbose=1
)

# 5. SIMPAN SEMUA RESOURCE (Agar terbaca di Streamlit)
print("💾 Menyimpan file untuk Streamlit...")

# Simpan Model (.h5)
model.save('model_stacked.h5')

# Simpan History (.pkl)
with open('history_stacked.pkl', 'wb') as f:
    pickle.dump(history.history, f)

# Simpan Data Test (Berisi X_test dan NILAI ASLI y_test)
with open('data_test.pkl', 'wb') as f:
    pickle.dump((X_test, y_test), f)

# Simpan Scaler (Opsional tapi penting untuk prediksi baru)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print(" SELESAI!")