import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score
)

import seaborn as sns

# =============================================================================
# 1. CARGA DE ARCHIVOS
# =============================================================================
eeg_df = pd.read_csv('imagenes/imagenes.csv', delimiter=',')
log_df = pd.read_csv('imagenes/log_imagenes.csv', delimiter=',')

sfreq = 250.0
canal_a_analizar = 'ch_7'  # Canal posterior (LPP)

eeg_crudo = eeg_df[canal_a_analizar].values
timestamps_eeg = eeg_df['lsl_timestamp'].values

# =============================================================================
# 2. FILTRADO TEMPORAL
# =============================================================================
def filtro_bandpass(signal, lowcut=1.0, highcut=20.0, fs=250.0, order=4):

    nyquist = 0.5 * fs

    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(order, [low, high], btype='band')

    return filtfilt(b, a, signal)

eeg_filtrado = filtro_bandpass(
    eeg_crudo,
    lowcut=1.0,
    highcut=20.0,
    fs=sfreq
)

# =============================================================================
# 3. SINCRONIZACIÓN
# =============================================================================
log_df.columns = log_df.columns.str.strip()

# Neutro vs Emocional
neutro_df = log_df[
    log_df['marker_code'] == 33027
]

emocional_df = log_df[
    (log_df['marker_code'] == 33025) |
    (log_df['marker_code'] == 33026)
]

lsl_inicio_eeg = timestamps_eeg[0]

lsl_neutros = (
    lsl_inicio_eeg +
    neutro_df['elapsed_experiment_time'].values
)

lsl_emocionales = (
    lsl_inicio_eeg +
    emocional_df['elapsed_experiment_time'].values
)

print(f"\nEEG arranca en LSL: {lsl_inicio_eeg:.3f}")
print(f"Imágenes neutras encontradas: {len(lsl_neutros)}")
print(f"Imágenes emocionales encontradas: {len(lsl_emocionales)}")

# =============================================================================
# 4. ÉPOCADO Y LIMPIEZA
# =============================================================================
pre_evento = int(0.2 * sfreq)
post_evento = int(0.8 * sfreq)

UMBRAL_RECHAZO = 35.0

def extraer_y_limpiar_epocas(eeg_signal, lista_lsl_targets):

    epocas_limpias = []
    rechazadas = 0

    for lsl_target in lista_lsl_targets:

        idx_evento = np.argmin(
            np.abs(timestamps_eeg - lsl_target)
        )

        if (
            idx_evento - pre_evento > 0 and
            idx_evento + post_evento < len(eeg_signal)
        ):

            fragmento = eeg_signal[
                idx_evento - pre_evento :
                idx_evento + post_evento
            ]

            # Baseline correction
            fragmento = (
                fragmento -
                np.mean(fragmento[:pre_evento])
            )

            # Rechazo de artefactos
            if np.max(np.abs(fragmento)) < UMBRAL_RECHAZO:
                epocas_limpias.append(fragmento)

            else:
                rechazadas += 1

    print(f"Limpias: {len(epocas_limpias)}")
    print(f"Rechazadas: {rechazadas}")

    return np.array(epocas_limpias)

print("\n--- IMÁGENES NEUTRAS ---")
epocas_neutras = extraer_y_limpiar_epocas(
    eeg_filtrado,
    lsl_neutros
)

print("\n--- IMÁGENES EMOCIONALES ---")
epocas_emocionales = extraer_y_limpiar_epocas(
    eeg_filtrado,
    lsl_emocionales
)

# =============================================================================
# 5. ERP PROMEDIO
# =============================================================================
promedio_neutro = np.mean(epocas_neutras, axis=0)
promedio_emocional = np.mean(epocas_emocionales, axis=0)

vector_tiempo = np.linspace(
    -200,
    800,
    len(promedio_neutro)
)

plt.figure(figsize=(11,4))

plt.plot(
    vector_tiempo,
    promedio_neutro,
    color='darkgray',
    linestyle='--',
    linewidth=2,
    label='Imágenes Neutras'
)

plt.plot(
    vector_tiempo,
    promedio_emocional,
    color='crimson',
    linewidth=2.5,
    label='Contenido Emocional'
)

plt.axvline(0, color='black', linestyle=':')

# Ventana clásica del LPP
plt.axvspan(
    350,
    700,
    color='pink',
    alpha=0.15,
    label='Ventana LPP'
)

plt.xlabel('Tiempo (ms)')
plt.ylabel('Voltaje ($\mu V$)')

plt.title(
    f'Potencial Evocado Emocional ({canal_a_analizar.upper()})'
)

plt.grid(True, linestyle=':')
plt.legend(loc='upper right')

plt.tight_layout()

plt.savefig(
    'erp_lpp_emociones.png',
    dpi=300
)

plt.show()

# =============================================================================
# 6. FEATURES TEMPORALES COMPLETAS
# =============================================================================
idx_inicio = pre_evento + int(0.35 * sfreq)
idx_fin = pre_evento + int(0.70 * sfreq)

def extraer_features_temporales(epocas):

    features = []

    for ep in epocas:

        ventana = ep[idx_inicio:idx_fin]

        # Conserva toda la dinámica temporal
        features.append(ventana)

    return np.array(features)

X_neutras = extraer_features_temporales(
    epocas_neutras
)

X_emocionales = extraer_features_temporales(
    epocas_emocionales
)

# Etiquetas
y_neutras = np.zeros(X_neutras.shape[0])
y_emocionales = np.ones(X_emocionales.shape[0])

X = np.vstack((X_neutras, X_emocionales))
y = np.concatenate((y_neutras, y_emocionales))

print("\nShape final del dataset:")
print(X.shape)

# =============================================================================
# 7. STRATIFIED KFOLD
# =============================================================================
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# =============================================================================
# 8. MODELO SVM
# =============================================================================
pipeline_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(
        kernel='linear',
        C=1.0,
        class_weight='balanced',
        random_state=42
    ))
])

y_pred_svm = cross_val_predict(
    pipeline_svm,
    X,
    y,
    cv=cv
)

print("\n==============================")
print("RESULTADOS SVM")
print("==============================")

print(classification_report(
    y,
    y_pred_svm,
    target_names=['Neutro', 'Emocional']
))

bal_acc_svm = balanced_accuracy_score(
    y,
    y_pred_svm
)

print(f"Balanced Accuracy SVM: {bal_acc_svm:.3f}")

# =============================================================================
# 9. RANDOM FOREST
# =============================================================================
modelo_rf = RandomForestClassifier(
    n_estimators=60,
    max_depth=3,
    class_weight='balanced',
    random_state=42
)

y_pred_rf = cross_val_predict(
    modelo_rf,
    X,
    y,
    cv=cv
)

print("\n==============================")
print("RESULTADOS RANDOM FOREST")
print("==============================")

print(classification_report(
    y,
    y_pred_rf,
    target_names=['Neutro', 'Emocional']
))

bal_acc_rf = balanced_accuracy_score(
    y,
    y_pred_rf
)

print(f"Balanced Accuracy RF: {bal_acc_rf:.3f}")

# =============================================================================
# 10. MATRICES DE CONFUSIÓN
# =============================================================================
fig, ax = plt.subplots(1, 2, figsize=(10,4.5))

TAMANO_NUMEROS = 18
TAMANO_ETIQUETAS = 15

# =============================================================================
# SVM
# =============================================================================
cm_svm = confusion_matrix(y, y_pred_svm)

sns.heatmap(
    cm_svm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Neutro', 'Emocional'],
    yticklabels=['Neutro', 'Emocional'],
    annot_kws={
        "size": TAMANO_NUMEROS,
        "weight": "bold"
    },
    ax=ax[0]
)

ax[0].set_title(
    'SVM',
    fontsize=TAMANO_ETIQUETAS + 2,
    weight='bold'
)

ax[0].tick_params(
    labelsize=TAMANO_ETIQUETAS - 1
)

# =============================================================================
# RANDOM FOREST
# =============================================================================
cm_rf = confusion_matrix(y, y_pred_rf)

sns.heatmap(
    cm_rf,
    annot=True,
    fmt='d',
    cmap='Oranges',
    xticklabels=['Neutro', 'Emocional'],
    yticklabels=['Neutro', 'Emocional'],
    annot_kws={
        "size": TAMANO_NUMEROS,
        "weight": "bold"
    },
    ax=ax[1]
)

ax[1].set_title(
    'Random Forest',
    fontsize=TAMANO_ETIQUETAS + 2,
    weight='bold'
)

ax[1].tick_params(
    labelsize=TAMANO_ETIQUETAS - 1
)

plt.tight_layout()

plt.savefig(
    'comparativa_matrices_imagenes.png',
    dpi=300
)

plt.show()