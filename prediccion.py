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
eeg_df = pd.read_csv('prediccion/prediccion.csv', delimiter=',')
log_df = pd.read_csv('prediccion/pred_events.csv', delimiter=',') 

sfreq = 250.0  # 250 Hz
canal_a_analizar = 'ch_1'  # CH_1 frontal/central es la posta para FRN

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

# Triggers numéricos: 33030 = Ganancia Esperada, 33033 = Pérdida Inesperada
ganancia_esperada_df = log_df[log_df['marker_code'] == 33030]
perdida_inesperada_df = log_df[log_df['marker_code'] == 33033]

lsl_inicio_eeg = timestamps_eeg[0]

lsl_ganancias = lsl_inicio_eeg + ganancia_esperada_df['elapsed_experiment_time'].values
lsl_perdidas = lsl_inicio_eeg + perdida_inesperada_df['elapsed_experiment_time'].values

print(f"\nEEG arranca en LSL: {lsl_inicio_eeg:.3f}")
print(f"Se encontraron {len(lsl_ganancias)} ensayos de Ganancia Esperada (33030).")
print(f"Se encontraron {len(lsl_perdidas)} ensayos de Pérdida Inesperada (33033).")

# =============================================================================
# 4. ÉPOCADO Y RECHAZO DE ARTEFACTOS
# =============================================================================
pre_evento = int(0.2 * sfreq)   
post_evento = int(0.8 * sfreq)  
UMBRAL_RECHAZO = 120.0          

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

print("\n--- Procesando Ganancias Esperadas ---")
epocas_ganancia = extraer_y_limpiar_epocas(eeg_filtrado, lsl_ganancias)

print("\n--- Procesando Pérdidas Inesperadas ---")
epocas_perdida = extraer_y_limpiar_epocas(eeg_filtrado, lsl_perdidas)

# Promedios para la curva
promedio_ganancia = np.mean(epocas_ganancia, axis=0)
promedio_perdida = np.mean(epocas_perdida, axis=0)

# =============================================================================
# 5. GRAFICAR ERP (Guarda la curva automática)
# =============================================================================
vector_tiempo = np.linspace(-200, 800, len(promedio_ganancia))
plt.figure(figsize=(11, 4))
plt.plot(vector_tiempo, promedio_ganancia, color='forestgreen', linewidth=2, label='Ganancia Esperada')
plt.plot(vector_tiempo, promedio_perdida, color='crimson', linewidth=2.5, label='Pérdida Inesperada')
plt.axvline(0, color='black', linestyle=':')
plt.axvspan(200, 300, color='red', alpha=0.08, label='Ventana FRN')
plt.axvspan(300, 500, color='gold', alpha=0.12, label='Ventana P300')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Voltaje Neto ($\mu V$)')
plt.title(f'ERP de Aprendizaje por Refuerzo ({canal_a_analizar.upper()}) - Caja Ganadora')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('erp_reinforcement_learning.png', dpi=300)
plt.show()
plt.close()

# =============================================================================
# 6. MACHINE LEARNING: EXTRACCIÓN DE FEATURES EN VENTANA DE FEEDBACK (200 a 500 ms)
# =============================================================================
idx_inicio = int((0.2 + 0.20) * sfreq) 
idx_fin = int((0.2 + 0.50) * sfreq)

def extraer_features_por_epoca(epocas):
    features = []
    for ep in epocas:
        ventana = ep[idx_inicio:idx_fin]
        v_min = np.min(ventana)      # Pozo de la FRN
        v_max = np.max(ventana)      # Cima de la P300
        v_mean = np.mean(ventana)    # Voltaje promedio neto
        v_std = np.std(ventana)      # Variabilidad morfológica
        features.append([v_min, v_max, v_mean, v_std])
    return np.array(features)

X_ganancias = extraer_features_por_epoca(epocas_ganancia)      
X_perdidas = extraer_features_por_epoca(epocas_perdida)

# Etiquetas: 0 = Ganancia, 1 = Pérdida
y_ganancias = np.zeros(X_ganancias.shape[0])
y_perdidas = np.ones(X_perdidas.shape[0])

X = np.vstack((X_ganancias, X_perdidas))
y = np.concatenate((y_ganancias, y_perdidas))

# División estratificada 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# =============================================================================
# 7. MODELO 1: SVM LINEAL
# =============================================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

modelo_svm = SVC(kernel='linear', C=1.0, class_weight='balanced', random_state=42)
modelo_svm.fit(X_train_scaled, y_train)
y_pred_svm = modelo_svm.predict(X_test_scaled)

print("\n=== REPORTE DE CLASIFICACIÓN (SVM - LEARNING) ===")
print(classification_report(y_test, y_pred_svm, target_names=['Ganancia', 'Pérdida']))

# =============================================================================
# 8. MODELO 2: RANDOM FOREST
# =============================================================================
modelo_rf = RandomForestClassifier(n_estimators=60, max_depth=3, class_weight='balanced', random_state=42)
modelo_rf.fit(X_train, y_train)
y_pred_rf = modelo_rf.predict(X_test)

print("\n=== REPORTE DE CLASIFICACIÓN (RANDOM FOREST - LEARNING) ===")
print(classification_report(y_test, y_pred_rf, target_names=['Ganancia', 'Pérdida']))

# =============================================================================
# 9. GRAFICAR MATRICES DE CONFUSIÓN COMPARATIVAS (NÚMEROS GRANDES - PREDICCIÓN)
# =============================================================================
fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))

# Configuración del tamaño de letra idéntica a tu bloque
TAMANO_NUMEROS = 18 
TAMANO_ETIQUETAS = 15

# Matriz 1: SVM (Paleta Verde con tus etiquetas de aprendizaje)
cm_svm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Ganancia', 'Pérdida'], yticklabels=['Ganancia', 'Pérdida'], 
            annot_kws={"size": TAMANO_NUMEROS, "weight": "bold"}, ax=ax[0])

ax[0].set_title('SVM', fontsize=TAMANO_ETIQUETAS + 2, weight='bold')
ax[0].tick_params(labelsize=TAMANO_ETIQUETAS - 1)

# Matriz 2: Random Forest (Paleta Naranja con tus etiquetas de aprendizaje)
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Oranges', 
            xticklabels=['Ganancia', 'Pérdida'], yticklabels=['Ganancia', 'Pérdida'], 
            annot_kws={"size": TAMANO_NUMEROS, "weight": "bold"}, ax=ax[1])

ax[1].set_title('Random Forest', fontsize=TAMANO_ETIQUETAS + 2, weight='bold')
ax[1].tick_params(labelsize=TAMANO_ETIQUETAS - 1)

plt.tight_layout()
plt.savefig('comparativa_matrices_learning.png', dpi=300)
plt.show()

# Extra: Importancia de características
importancias = modelo_rf.feature_importances_
print("\n--- Importancia de las Características (Random Forest) ---")
print(f"Voltaje Mínimo (v_min - FRN): {importancias[0]:.3f}")
print(f"Voltaje Máximo (v_max - P300): {importancias[1]:.3f}")
print(f"Voltaje Promedio (v_mean): {importancias[2]:.3f}")
print(f"Desviación Estándar (v_std): {importancias[3]:.3f}")