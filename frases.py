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
eeg_df = pd.read_csv('frases/frases.csv', delimiter=',')
log_df = pd.read_csv('frases/frases_beh.csv', delimiter=',') 

sfreq = 250.0  # 250 Hz
canal_a_analizar = 'ch_1'  # CH_1 frontal (ideal para N400 semántica)

eeg_crudo = eeg_df[canal_a_analizar].values
timestamps_eeg = eeg_df['lsl_timestamp'].values

# =============================================================================
# 2. FILTRADO TEMPORAL (Pasa-Altos de 1 Hz)
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

# Separamos usando los códigos correctos del log de frases
normales_df = log_df[log_df['critical_marker_code'] == 33026]
absurdas_df = log_df[log_df['critical_marker_code'] == 33027]

lsl_inicio_eeg = timestamps_eeg[0]

lsl_normales = lsl_inicio_eeg + normales_df['elapsed_experiment_time'].values
lsl_absurdas = lsl_inicio_eeg + absurdas_df['elapsed_experiment_time'].values

print(f"\nEEG arranca en LSL: {lsl_inicio_eeg:.3f}")
print(f"Se encontraron {len(lsl_normales)} frases Normales (33026).")
print(f"Se encontraron {len(lsl_absurdas)} frases Absurdas (33027).")

# =============================================================================
# 4. ÉPOCADO Y RECHAZO DE ARTEFACTOS
# =============================================================================
pre_evento = int(0.2 * sfreq)   # 200 ms antes
post_evento = int(0.8 * sfreq)  # 800 ms después
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

print("\n--- Procesando Condición Normal ---")
epocas_normales = extraer_y_limpiar_epocas(eeg_filtrado, lsl_normales)

print("\n--- Procesando Condición Absurda ---")
epocas_absurdas = extraer_y_limpiar_epocas(eeg_filtrado, lsl_absurdas)

# Promedios para graficar las ondas del ERP
promedio_normal = np.mean(epocas_normales, axis=0)
promedio_absurda = np.mean(epocas_absurdas, axis=0)

# =============================================================================
# 5. GRAFICAR ERP
# =============================================================================
vector_tiempo = np.linspace(-200, 800, len(promedio_normal))
plt.figure(figsize=(11, 4))
plt.plot(vector_tiempo, promedio_normal, color='teal', linewidth=2, label='Condición Normal')
plt.plot(vector_tiempo, promedio_absurda, color='darkorange', linewidth=2.5, label='Condición Absurda')
plt.axvline(0, color='black', linestyle=':')

# Ajustado finamente para la velocidad del procesamiento semántico del lenguaje
plt.axvspan(350, 550, color='cyan', alpha=0.08, label='Ventana N400 Semántica')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Voltaje Neto ($\mu V$)')
plt.title(f'Potencial Evocado Relacionado a Lenguaje ({canal_a_analizar.upper()}) - Efecto N400')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('erp_n400_linguistica.png', dpi=300)
plt.show()
plt.close()

# =============================================================================
# 6. MACHINE LEARNING: EXTRACCIÓN DE FEATURES EN VENTANA SEMÁNTICA (350 a 550 ms)
# =============================================================================
idx_inicio = pre_evento + int(0.35 * sfreq)  # Índice 137 (350 ms)
idx_fin = pre_evento + int(0.55 * sfreq)     # Índice 187 (550 ms)

def extraer_features_por_epoca(epocas):
    features = []
    for ep in epocas:
        ventana = ep[idx_inicio:idx_fin]
        v_max = np.max(ventana)      # pico del valle N400
        v_min = np.min(ventana)      # Fondo del valle N400
        v_mean = np.mean(ventana)    # Voltaje promedio integrado
        v_std = np.std(ventana)      # Varianza morfológica
        features.append([v_min, v_mean, v_std])
    return np.array(features)

X_normales = extraer_features_por_epoca(epocas_normales)      
X_absurdas = extraer_features_por_epoca(epocas_absurdas)

# Etiquetas: 0 = Normal, 1 = Absurda
y_normales = np.zeros(X_normales.shape[0])
y_absurdas = np.ones(X_absurdas.shape[0])

X = np.vstack((X_normales, X_absurdas))
y = np.concatenate((y_normales, y_absurdas))

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

print("\n=== REPORTE DE CLASIFICACIÓN (SVM - FRASES) ===")
print(classification_report(y_test, y_pred_svm, target_names=['Normal', 'Absurda']))

# =============================================================================
# 8. MODELO 2: RANDOM FOREST
# =============================================================================
modelo_rf = RandomForestClassifier(n_estimators=60, max_depth=3, class_weight='balanced', random_state=42)
modelo_rf.fit(X_train, y_train)
y_pred_rf = modelo_rf.predict(X_test)

print("\n=== REPORTE DE CLASIFICACIÓN (RANDOM FOREST - FRASES) ===")
print(classification_report(y_test, y_pred_rf, target_names=['Normal', 'Absurda']))

# =============================================================================
# 9. GRAFICAR MATRICES DE CONFUSIÓN COMPARATIVAS (NÚMEROS GRANDES - FRASES)
# =============================================================================
fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))

# Configuración del tamaño de letra idéntica a tu bloque
TAMANO_NUMEROS = 18 
TAMANO_ETIQUETAS = 15

# Matriz 1: SVM (Paleta Azul con tus etiquetas lingüísticas)
cm_svm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Absurda'], yticklabels=['Normal', 'Absurda'], 
            annot_kws={"size": TAMANO_NUMEROS, "weight": "bold"}, ax=ax[0])

ax[0].set_title('SVM', fontsize=TAMANO_ETIQUETAS + 2, weight='bold')
ax[0].tick_params(labelsize=TAMANO_ETIQUETAS - 1)

# Matriz 2: Random Forest (Paleta Violeta con tus etiquetas lingüísticas)
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Purples', 
            xticklabels=['Normal', 'Absurda'], yticklabels=['Normal', 'Absurda'], 
            annot_kws={"size": TAMANO_NUMEROS, "weight": "bold"}, ax=ax[1])

ax[1].set_title('Random Forest', fontsize=TAMANO_ETIQUETAS + 2, weight='bold')
ax[1].tick_params(labelsize=TAMANO_ETIQUETAS - 1)

plt.tight_layout()
plt.savefig('comparativa_matrices_frases.png', dpi=300)
plt.show()

# Extra: Importancia de características
importancias = modelo_rf.feature_importances_
print("\n--- Importancia de las Características (Random Forest - Frases) ---")
print(f"Voltaje Mínimo (v_max): {importancias[0]:.3f}")
print(f"Voltaje Mínimo (v_min): {importancias[1]:.3f}")
print(f"Voltaje Promedio (v_mean): {importancias[2]:.3f}")
print(f"Desviación Estándar (v_std): {importancias[3]:.3f}")