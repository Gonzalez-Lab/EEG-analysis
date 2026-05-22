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
eeg_df = pd.read_csv('imagenes/imagenes.csv', delimiter=',')
log_df = pd.read_csv('imagenes/log_imagenes.csv', delimiter=',') 

sfreq = 250.0  # 250 Hz
canal_a_analizar = 'ch_7'  # Canal parietal posterior (ideal para registrar el componente LPP)

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

# Condición Control (Neutro = 33027) vs Contenido Afectivo (Feliz = 33025 o Triste = 33026)
neutro_df = log_df[log_df['marker_code'] == 33027]
emocional_df = log_df[(log_df['marker_code'] == 33025) | (log_df['marker_code'] == 33026)]

lsl_inicio_eeg = timestamps_eeg[0]

lsl_neutros = lsl_inicio_eeg + neutro_df['elapsed_experiment_time'].values
lsl_emocionales = lsl_inicio_eeg + emocional_df['elapsed_experiment_time'].values

print(f"\nEEG arranca en LSL: {lsl_inicio_eeg:.3f}")
print(f"Se encontraron {len(lsl_neutros)} imágenes Neutras.")
print(f"Se encontraron {len(lsl_emocionales)} imágenes Emocionales.")

# =============================================================================
# 4. ÉPOCADO LARGO Y RECHAZO DE ARTEFACTOS
# =============================================================================
pre_evento = int(0.2 * sfreq)   # 50 muestras (200 ms antes)
post_evento = int(0.8 * sfreq)  # 200 muestras (800 ms después)
UMBRAL_RECHAZO = 50.0          

def extraer_y_limpiar_epocas(eeg_signal, lista_lsl_targets):
    epocas_limpias = []
    ensayos_rechazados = 0
    
    for lsl_target in lista_lsl_targets:
        idx_evento = np.argmin(np.abs(timestamps_eeg - lsl_target))
        
        if idx_evento - pre_evento > 0 and idx_evento + post_evento < len(eeg_signal):
            fragmento = eeg_signal[idx_evento - pre_evento : idx_evento + post_evento]
            fragmento_centrado = fragmento - np.mean(fragmento[:pre_evento])
            
            # Control estricto por umbral para remoción de parpadeos
            if np.max(np.abs(fragmento_centrado)) < UMBRAL_RECHAZO:
                epocas_limpias.append(fragmento_centrado)
            else:
                ensayos_rechazados += 1
                
    print(f"-> Procesados: {len(epocas_limpias)} épocas limpias. Rechazadas: {ensayos_rechazados}")
    return np.array(epocas_limpias)

print("\n--- Procesando Imágenes Neutras ---")
epocas_neutras = extraer_y_limpiar_epocas(eeg_filtrado, lsl_neutros)

print("\n--- Procesando Imágenes Emocionales ---")
epocas_emocionales = extraer_y_limpiar_epocas(eeg_filtrado, lsl_emocionales)

# Promedios para el ERP
promedio_neutro = np.mean(epocas_neutras, axis=0)
promedio_emocional = np.mean(epocas_emocionales, axis=0)

# =============================================================================
# 5. GRAFICAR ERP
# =============================================================================
vector_tiempo = np.linspace(-200, 800, len(promedio_neutro))
plt.figure(figsize=(11, 4))
plt.plot(vector_tiempo, promedio_neutro, color='darkgray', linestyle='--', linewidth=2, label='Imágenes Neutras')
plt.plot(vector_tiempo, promedio_emocional, color='crimson', linewidth=2.5, label='Contenido Emocional')
plt.axvline(0, color='black', linestyle=':')

# Sombreamos la ventana de activación sostenida del LPP (350ms a 700ms)
plt.axvspan(350, 700, color='pink', alpha=0.15, label='Ventana LPP Emocional')
plt.xlabel('Tiempo desde la presentación de la imagen (ms)')
plt.ylabel('Voltaje Neto ($\mu V$)')
plt.title(f'Potencial Evocado Sostenido ({canal_a_analizar.upper()}) - Procesamiento Afectivo')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('erp_lpp_emociones.png', dpi=300)
plt.show()
plt.close()

# =============================================================================
# 6. MACHINE LEARNING: EXTRACCIÓN DE FEATURES EN VENTANA LPP (350 a 700 ms)
# =============================================================================
idx_inicio = pre_evento + int(0.35 * sfreq)  # Índice 137 (350 ms)
idx_fin = pre_evento + int(0.70 * sfreq)     # Índice 225 (700 ms)

def extraer_features_por_epoca(epocas):
    features = []
    for ep in epocas:
        ventana = ep[idx_inicio:idx_fin]
        v_max = np.max(ventana)      # Cambiado a MÁXIMO porque el LPP es una deflexión positiva
        v_min = np.min(ventana)      # Voltaje minimo
        v_mean = np.mean(ventana)    # Voltaje promedio integrado de la ventana
        v_std = np.std(ventana)      # Variabilidad morfológica tónica
        features.append([v_max, v_mean, v_std])
    return np.array(features)

X_neutras = extraer_features_por_epoca(epocas_neutras)      
X_emocionales = extraer_features_por_epoca(epocas_emocionales)

# Etiquetas: 0 = Neutro, 1 = Emocional
y_neutras = np.zeros(X_neutras.shape[0])
y_emocionales = np.ones(X_emocionales.shape[0])

X = np.vstack((X_neutras, X_emocionales))
y = np.concatenate((y_neutras, y_emocionales))

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

print("\n=== REPORTE DE CLASIFICACIÓN (SVM - IMÁGENES) ===")
print(classification_report(y_test, y_pred_svm, target_names=['Neutro', 'Emocional']))

# =============================================================================
# 8. MODELO 2: RANDOM FOREST
# =============================================================================
modelo_rf = RandomForestClassifier(n_estimators=60, max_depth=3, class_weight='balanced', random_state=42)
modelo_rf.fit(X_train, y_train)
y_pred_rf = modelo_rf.predict(X_test)

print("\n=== REPORTE DE CLASIFICACIÓN (RANDOM FOREST - IMÁGENES) ===")
print(classification_report(y_test, y_pred_rf, target_names=['Neutro', 'Emocional']))

# =============================================================================
# 9. GRAFICAR MATRICES DE CONFUSIÓN COMPARATIVAS (NÚMEROS GRANDES - IMÁGENES)
# =============================================================================
fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))

# Configuración del tamaño de letra idéntica a tu bloque
TAMANO_NUMEROS = 18 
TAMANO_ETIQUETAS = 15

# Matriz 1: SVM (Paleta Azul con tus etiquetas afectivas)
cm_svm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Neutro', 'Emocional'], yticklabels=['Neutro', 'Emocional'], 
            annot_kws={"size": TAMANO_NUMEROS, "weight": "bold"}, ax=ax[0])

ax[0].set_title('SVM', fontsize=TAMANO_ETIQUETAS + 2, weight='bold')
ax[0].tick_params(labelsize=TAMANO_ETIQUETAS - 1)

# Matriz 2: Random Forest (Paleta Naranja con tus etiquetas afectivas)
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Oranges', 
            xticklabels=['Neutro', 'Emocional'], yticklabels=['Neutro', 'Emocional'], 
            annot_kws={"size": TAMANO_NUMEROS, "weight": "bold"}, ax=ax[1])

ax[1].set_title('Random Forest', fontsize=TAMANO_ETIQUETAS + 2, weight='bold')
ax[1].tick_params(labelsize=TAMANO_ETIQUETAS - 1)

plt.tight_layout()
plt.savefig('comparativa_matrices_imagenes.png', dpi=300)
plt.show()

# Extra: Importancia de características
importancias = modelo_rf.feature_importances_
print("\n--- Importancia de las Características (Random Forest - Imágenes) ---")
print(f"Voltaje Máximo (v_max): {importancias[0]:.3f}")
print(f"Voltaje Minimo (v_min): {importancias[1]:.3f}")
print(f"Voltaje Promedio (v_mean): {importancias[2]:.3f}")
print(f"Desviación Estándar (v_std): {importancias[3]:.3f}")