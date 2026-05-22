import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# =============================================================================
# 1. CARGA DE ARCHIVOS
# =============================================================================
eeg_df = pd.read_csv('clubes/clubes.csv', delimiter=',')
log_df = pd.read_csv('clubes/log_clubes.csv', delimiter=',')

sfreq = 250.0  # Frecuencia de muestreo
canal_a_analizar = 'ch_7'  # Canal parietal posterior (ideal para P300)

eeg_crudo = eeg_df[canal_a_analizar].values
timestamps_eeg = eeg_df['lsl_timestamp'].values

# =============================================================================
# 2. FILTRADO TEMPORAL
# =============================================================================
def filtro_pasa_altos(signal, cutoff=1.0, fs=250.0, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, signal)

eeg_filtrado = filtro_pasa_altos(eeg_crudo, cutoff=1.0, fs=sfreq)

# =============================================================================
# 3. EXTRACCIÓN AUTOMÁTICA Y SINCRONIZACIÓN
# =============================================================================
log_df.columns = log_df.columns.str.strip()

# 33026 33029 = Estímulo Estándar, 33027 33030= Estímulo Desviado (Objeto Raro)
standards_df = log_df[(log_df['marker_code'] == 33026) | (log_df['marker_code'] == 33029)]
deviants_df = log_df[(log_df['marker_code'] == 33027) | (log_df['marker_code'] == 33030)]

lsl_inicio_eeg = timestamps_eeg[0] 

lsl_standards = lsl_inicio_eeg + standards_df['elapsed_experiment_time'].values
lsl_deviants = lsl_inicio_eeg + deviants_df['elapsed_experiment_time'].values

print(f"\nEEG arranca en LSL: {lsl_inicio_eeg:.3f}")
print(f"Se encontraron {len(lsl_standards)} estímulos Estándar.")
print(f"Se encontraron {len(lsl_deviants)} estímulos Desviados.")

# =============================================================================
# 4. ÉPOCADO Y RECHAZO DE ARTEFACTOS
# =============================================================================
pre_evento = int(0.2 * sfreq)   # 50 muestras
post_evento = int(0.8 * sfreq)  # 200 muestras
UMBRAL_RECHAZO = 50.0

def extraer_y_limpiar_epocas(eeg_signal, lista_lsl_targets):
    epocas_limpias = []
    ensayos_rechazados = 0
    
    for lsl_target in lista_lsl_targets:
        idx_evento = np.argmin(np.abs(timestamps_eeg - lsl_target))
        
        if idx_evento - pre_evento > 0 and idx_evento + post_evento < len(eeg_signal):
            fragmento = eeg_signal[idx_evento - pre_evento : idx_evento + post_evento]
            fragmento_centrado = fragmento - np.mean(fragmento[:pre_evento])
            
            if np.max(np.abs(fragmento_centrado)) < UMBRAL_RECHAZO:
                epocas_limpias.append(fragmento_centrado)
            else:
                ensayos_rechazados += 1
                
    print(f"-> Procesados: {len(epocas_limpias)} épocas limpias. Rechazadas: {ensayos_rechazados}")
    return np.array(epocas_limpias)

print("\n--- Procesando Estímulos Estándar ---")
epocas_standard = extraer_y_limpiar_epocas(eeg_filtrado, lsl_standards)

print("\n--- Procesando Estímulos Desviados (Sorpresa) ---")
epocas_deviant = extraer_y_limpiar_epocas(eeg_filtrado, lsl_deviants)

promedio_standard = np.mean(epocas_standard, axis=0)
promedio_deviant = np.mean(epocas_deviant, axis=0)

# =============================================================================
# 5. GRAFICAR ERP (Guarda la curva automática)
# =============================================================================
vector_tiempo = np.linspace(-200, 800, len(promedio_standard))
plt.figure(figsize=(11, 4))
plt.plot(vector_tiempo, promedio_standard, color='gray', linestyle='--', linewidth=2, label='Estándar (Clubes)')
plt.plot(vector_tiempo, promedio_deviant, color='blue', linewidth=2.5, label='Desviado (Objeto Raro)')
plt.axvline(0, color='black', linestyle=':')
plt.axvspan(300, 500, color='gold', alpha=0.15, label='Ventana P300')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Voltaje Neto ($\mu V$)')
plt.title(f'ERP Limpio en {canal_a_analizar.upper()} (Filtrado + Rechazo de Parpadeos)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('erp_oddball.png', dpi=300)
plt.show()
plt.close()

# =============================================================================
# 6. MACHINE LEARNING: EXTRACCIÓN DE FEATURES EN VENTANA P300 (300 a 500 ms)
# =============================================================================
idx_inicio = pre_evento + int(0.30 * sfreq)  # Índice 125 (300 ms)
idx_fin = pre_evento + int(0.50 * sfreq)     # Índice 175 (500 ms)

def extraer_features_por_epoca(epocas):
    features = []
    for ep in epocas:
        ventana = ep[idx_inicio:idx_fin]
        v_max = np.max(ventana)      # Cima de la P300
        v_min = np.min(ventana)      # fondo de la P300
        v_mean = np.mean(ventana)    # Voltaje promedio neto
        v_std = np.std(ventana)      # Variabilidad de la curva
        features.append([v_max, v_min, v_mean, v_std])
    return np.array(features)

X_standards = extraer_features_por_epoca(epocas_standard)      
X_deviants = extraer_features_por_epoca(epocas_deviant)

# Etiquetas: 0 = Estándar, 1 = Desviado
y_standards = np.zeros(X_standards.shape[0])
y_deviants = np.ones(X_deviants.shape[0])

X = np.vstack((X_standards, X_deviants))
y = np.concatenate((y_standards, y_deviants))

# División estratificada 75/25
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# =============================================================================
# 7. MODELO 1: SVM LINEAL
# =============================================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

modelo_svm = SVC(kernel='linear', C=1.0, class_weight='balanced', random_state=42)
modelo_svm.fit(X_train_scaled, y_train)
y_pred_svm = modelo_svm.predict(X_test_scaled)

print("\n=== REPORTE DE CLASIFICACIÓN (SVM - CLUBES) ===")
print(classification_report(y_test, y_pred_svm, target_names=['Estándar', 'Desviado']))

# =============================================================================
# 8. MODELO 2: RANDOM FOREST
# =============================================================================
modelo_rf = RandomForestClassifier(n_estimators=60, max_depth=3, class_weight='balanced', random_state=42)
modelo_rf.fit(X_train, y_train)
y_pred_rf = modelo_rf.predict(X_test)

print("\n=== REPORTE DE CLASIFICACIÓN (RANDOM FOREST - CLUBES) ===")
print(classification_report(y_test, y_pred_rf, target_names=['Estándar', 'Desviado']))

# =============================================================================
# 9. GRAFICAR MATRICES DE CONFUSIÓN COMPARATIVAS (NÚMEROS GRANDES - ODDBALL)
# =============================================================================
fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))

# Configuración del tamaño de letra idéntica a tu bloque
TAMANO_NUMEROS = 18 
TAMANO_ETIQUETAS = 15

# Matriz 1: SVM (Paleta Azul con tus etiquetas de oddball)
cm_svm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Estándar', 'Desviado'], yticklabels=['Estándar', 'Desviado'], 
            annot_kws={"size": TAMANO_NUMEROS, "weight": "bold"}, ax=ax[0])

ax[0].set_title('SVM', fontsize=TAMANO_ETIQUETAS + 2, weight='bold')
ax[0].tick_params(labelsize=TAMANO_ETIQUETAS - 1)

# Matriz 2: Random Forest (Paleta Azul-Verdosa con tus etiquetas de oddball)
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='YlGnBu', 
            xticklabels=['Estándar', 'Desviado'], yticklabels=['Estándar', 'Desviado'], 
            annot_kws={"size": TAMANO_NUMEROS, "weight": "bold"}, ax=ax[1])

ax[1].set_title('Random Forest', fontsize=TAMANO_ETIQUETAS + 2, weight='bold')
ax[1].tick_params(labelsize=TAMANO_ETIQUETAS - 1)

plt.tight_layout()
plt.savefig('comparativa_matrices_clubes.png', dpi=300)
plt.show()

# Extra: Importancia de características
importancias = modelo_rf.feature_importances_
print("\n--- Importancia de las Características (Random Forest) ---")
print(f"Voltaje Máximo (v_max - P300): {importancias[0]:.3f}")
print(f"Voltaje Minimo (v_min - P300): {importancias[0]:.3f}")
print(f"Voltaje Promedio (v_mean): {importancias[1]:.3f}")
print(f"Desviación Estándar (v_std): {importancias[2]:.3f}")

