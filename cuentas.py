#####
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
eeg_df = pd.read_csv('cuentas/cuentas.csv', delimiter=',')
log_df = pd.read_csv('cuentas/cuentas_events.csv', delimiter=',') 

sfreq = 250.0  # 250 Hz
canal_a_analizar = 'ch_1'  # CH_1 frontal/central para el procesamiento del error lógico

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

# Agrupamos operaciones Correctas (33025 y 33027) vs Incorrectas/Conflicto (33026 y 33028)
correctas_df = log_df[(log_df['marker_code'] == 33025) | (log_df['marker_code'] == 33027)]
incorrectas_df = log_df[(log_df['marker_code'] == 33026) | (log_df['marker_code'] == 33028)]

lsl_inicio_eeg = timestamps_eeg[0]

lsl_correctas = lsl_inicio_eeg + correctas_df['elapsed_experiment_time'].values
lsl_incorrectas = lsl_inicio_eeg + incorrectas_df['elapsed_experiment_time'].values

print(f"\nEEG arranca en LSL: {lsl_inicio_eeg:.3f}")
print(f"Se encontraron {len(lsl_correctas)} operaciones Correctas.")
print(f"Se encontraron {len(lsl_incorrectas)} operaciones Incorrectas (Conflicto).")

# =============================================================================
# 4. ÉPOCADO Y RECHAZO DE ARTEFACTOS
# =============================================================================
pre_evento = int(0.2 * sfreq)   # 50 muestras (200 ms)
post_evento = int(0.8 * sfreq)  # 200 muestras (800 ms)
UMBRAL_RECHAZO = 50.0          

def extraer_y_limpiar_epocas(eeg_signal, lista_lsl_targets):
    epocas_limpias = []
    ensayos_rechazados = 0
    
    for lsl_target in lista_lsl_targets:
        idx_evento = np.argmin(np.abs(timestamps_eeg - lsl_target))
        
        if idx_evento - pre_evento > 0 and idx_evento + post_evento < len(eeg_signal):
            fragmento = eeg_signal[idx_evento - pre_evento : idx_evento + post_evento]
            fragmento_centrado = fragmento - np.mean(fragmento[:pre_evento])
            
            # Control por umbral estricto para remover parpadeos
            if np.max(np.abs(fragmento_centrado)) < UMBRAL_RECHAZO:
                epocas_limpias.append(fragmento_centrado)
            else:
                ensayos_rechazados += 1
                
    print(f"-> Procesados: {len(epocas_limpias)} épocas limpias. Rechazadas: {ensayos_rechazados}")
    return np.array(epocas_limpias)

print("\n--- Procesando Operaciones Correctas ---")
epocas_corr = extraer_y_limpiar_epocas(eeg_filtrado, lsl_correctas)

print("\n--- Procesando Operaciones Incorrectas ---")
epocas_incorr = extraer_y_limpiar_epocas(eeg_filtrado, lsl_incorrectas)

# Promedios para graficar las ondas
promedio_corr = np.mean(epocas_corr, axis=0)
promedio_incorr = np.mean(epocas_incorr, axis=0)

# =============================================================================
# 5. GRAFICAR ERP
# =============================================================================
vector_tiempo = np.linspace(-200, 800, len(promedio_corr))
plt.figure(figsize=(11, 4))
plt.plot(vector_tiempo, promedio_corr, color='navy', linewidth=2, label='Ecuaciones Correctas')
plt.plot(vector_tiempo, promedio_incorr, color='darkred', linewidth=2.5, label='Ecuaciones Incorrectas (Mismatch)')
plt.axvline(0, color='black', linestyle=':')

# Ajuste fino visual: Ventana de evaluación del conflicto numérico (250ms a 450ms)
plt.axvspan(250, 450, color='orange', alpha=0.1, label='Ventana N400 Matemática')
plt.xlabel('Tiempo desde la aparición de la ecuación (ms)')
plt.ylabel('Voltaje Neto ($\mu V$)')
plt.title(f'Potencial Evocado Relacionado a Conflicto ({canal_a_analizar.upper()}) - Aritmética Mental')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('erp_aritmetica_error.png', dpi=300)
plt.show()
plt.close()

# =============================================================================
# 6. MACHINE LEARNING: EXTRACCIÓN DE FEATURES EN VENTANA ARITMÉTICA (250 a 450 ms)
# =============================================================================
idx_inicio = pre_evento + int(0.25 * sfreq)  # Índice 112 (250 ms)
idx_fin = pre_evento + int(0.45 * sfreq)     # Índice 162 (450 ms)

def extraer_features_por_epoca(epocas):
    features = []
    for ep in epocas:
        ventana = ep[idx_inicio:idx_fin]
        v_max = np.max(ventana)      # Captura el maximo del desplome negativo
        v_min = np.min(ventana)      # Captura la profundidad del desplome negativo
        v_mean = np.mean(ventana)    # Voltaje promedio integrado
        v_std = np.std(ventana)      # Desviación estándar (morfología)
        features.append([v_min, v_mean, v_std])
    return np.array(features)

X_correctas = extraer_features_por_epoca(epocas_corr)      
X_incorrectas = extraer_features_por_epoca(epocas_incorr)

# Etiquetas: 0 = Correcto, 1 = Incorrecto/Conflicto
y_correctas = np.zeros(X_correctas.shape[0])
y_incorrectas = np.ones(X_incorrectas.shape[0])

X = np.vstack((X_correctas, X_incorrectas))
y = np.concatenate((y_correctas, y_incorrectas))

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

print("\n=== REPORTE DE CLASIFICACIÓN (SVM - ARITMÉTICA) ===")
print(classification_report(y_test, y_pred_svm, target_names=['Correcto', 'Incorrecto']))

# =============================================================================
# 8. MODELO 2: RANDOM FOREST
# =============================================================================
modelo_rf = RandomForestClassifier(n_estimators=60, max_depth=3, class_weight='balanced', random_state=42)
modelo_rf.fit(X_train, y_train)
y_pred_rf = modelo_rf.predict(X_test)

print("\n=== REPORTE DE CLASIFICACIÓN (RANDOM FOREST - ARITMÉTICA) ===")
print(classification_report(y_test, y_pred_rf, target_names=['Correcto', 'Incorrecto']))

# =============================================================================
# 9. GRAFICAR MATRICES DE CONFUSIÓN COMPARATIVAS (NÚMEROS GRANDES - CUENTAS)
# =============================================================================
fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))

# Configuración del tamaño de letra para que se lea perfecto en el PDF
TAMANO_NUMEROS = 18 
TAMANO_ETIQUETAS = 15

# Matriz 1: SVM (Paleta Azul)
cm_svm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Correcto', 'Incorrecto'], yticklabels=['Correcto', 'Incorrecto'], 
            annot_kws={"size": TAMANO_NUMEROS, "weight": "bold"}, ax=ax[0])

ax[0].set_title('SVM', fontsize=TAMANO_ETIQUETAS + 2, weight='bold')
ax[0].tick_params(labelsize=TAMANO_ETIQUETAS - 1)

# Matriz 2: Random Forest (Paleta Roja)
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Reds', 
            xticklabels=['Correcto', 'Incorrecto'], yticklabels=['Correcto', 'Incorrecto'], 
            annot_kws={"size": TAMANO_NUMEROS, "weight": "bold"}, ax=ax[1])

ax[1].set_title('Random Forest', fontsize=TAMANO_ETIQUETAS + 2, weight='bold')
ax[1].tick_params(labelsize=TAMANO_ETIQUETAS - 1)

plt.tight_layout()
plt.savefig('comparativa_matrices_cuentas.png', dpi=300)
plt.show()

# Extra: Importancia de características
importancias = modelo_rf.feature_importances_
print("\n--- Importancia de las Características (Random Forest - Cuentas) ---")
print(f"Voltaje Maximo (v_max): {importancias[0]:.3f}")
print(f"Voltaje Mínimo (v_min): {importancias[1]:.3f}")
print(f"Voltaje Promedio (v_mean): {importancias[2]:.3f}")
print(f"Desviación Estándar (v_std): {importancias[3]:.3f}")
# %%
