import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# Configuración técnica común
sfreq = 250.0  # 250 Hz
puntos_recorte = 4500  # Mantengo tus 18 segundos de recorte inicial
canal_indice = 8  # ch_1

# ==========================================
# 1. PROCESAR DATASET: OJOS CERRADOS
# ==========================================
signals_cerrados = pd.read_csv('ojoscerrados/ojoscerrados.csv', delimiter=',')
df_cerrados_limpio = signals_cerrados.iloc[puntos_recorte:].reset_index(drop=True)

eeg_cerrados = df_cerrados_limpio.values[:, canal_indice]
eeg_cerrados = eeg_cerrados - np.mean(eeg_cerrados)  # Centrar en 0

# FFT Ojos Cerrados
frecuencias_cerrados, potencia_cerrados = welch(eeg_cerrados, fs=sfreq, nperseg=1024)

# Encontrar pico Alpha para Ojos Cerrados
indices_alpha = np.where((frecuencias_cerrados >= 8) & (frecuencias_cerrados <= 12))[0]
frecuencias_alpha = frecuencias_cerrados[indices_alpha]
potencia_alpha_cerrados = potencia_cerrados[indices_alpha]

indice_max_alpha = np.argmax(potencia_alpha_cerrados)
frecuencia_pico_cerrados = frecuencias_alpha[indice_max_alpha]
potencia_pico_cerrados = potencia_alpha_cerrados[indice_max_alpha]


# ==========================================
# 2. PROCESAR DATASET: BASELINE (OJOS ABIERTOS)
# ==========================================
# Ajustá la ruta si tu archivo se llama distinto (ej. 'blinking/baseline.csv')
signals_baseline = pd.read_csv('baseline/baseline.csv', delimiter=',') 
df_baseline_limpio = signals_baseline.iloc[puntos_recorte:].reset_index(drop=True)

eeg_baseline = df_baseline_limpio.values[:, canal_indice]
eeg_baseline = eeg_baseline - np.mean(eeg_baseline)  # Centrar en 0

# FFT Baseline
frecuencias_baseline, potencia_baseline = welch(eeg_baseline, fs=sfreq, nperseg=1024)


# ==========================================
# 3. IMPRIMIR RESULTADOS EN CONSOLA
# ==========================================
print("-" * 50)
print(f"¡ANÁLISIS COMPARATIVO COMPLETADO!")
print(f"Pico Alpha detectado (Ojos Cerrados): {frecuencia_pico_cerrados:.2f} Hz")
print(f"Potencia del pico (Ojos Cerrados): {potencia_pico_cerrados:.4f} uV^2/Hz")
print("-" * 50)


# ==========================================
# 4. GRAFICAR AMBAS SEÑALES JUNTAS (CON ZOOM CORREGIDO)
# ==========================================
plt.figure(figsize=(12, 6))

# Graficamos las curvas
plt.plot(frecuencias_baseline, potencia_baseline, color='steelblue', alpha=0.7, 
         linewidth=2, label='Baseline (Ojos Abiertos)')
plt.plot(frecuencias_cerrados, potencia_cerrados, color='crimson', 
         linewidth=2.5, label='Ojos Cerrados')

# Resaltar la banda Alpha (8-12 Hz)
plt.axvspan(8, 12, color='orange', alpha=0.15, label='Banda Alpha (8-12 Hz)')

# Marcar el punto del pico de ojos cerrados
plt.plot(frecuencia_pico_cerrados, potencia_pico_cerrados, 'ro', markersize=9, 
         label=f'Pico Alpha Máx: {frecuencia_pico_cerrados:.2f} Hz')

# ------------------------------------------------------------------
# ¡ZONA DE CORRECCIÓN DE ESCALA!
# Forzamos el eje X a arrancar en 2 Hz para ignorar el ruido de continua
plt.xlim([2, 30])

# Calculamos el máximo de potencia PERO solo buscando en el rango de 2 a 30 Hz
# Esto evita que el ruido de 0-1 Hz nos rompa la escala del eje Y
mask_f_cerrados = (frecuencias_cerrados >= 2) & (frecuencias_cerrados <= 30)
mask_f_baseline = (frecuencias_baseline >= 2) & (frecuencias_baseline <= 30)

max_potencia_visible = max(max(potencia_cerrados[mask_f_cerrados]), 
                           max(potencia_baseline[mask_f_baseline]))

# Le damos un 10% de aire arriba del pico más alto
plt.ylim([0, max_potencia_visible * 1.1])
# ------------------------------------------------------------------

# Etiquetas y diseño
plt.xlabel('Frecuencia (Hz)', fontsize=12)
plt.ylabel('Densidad Espectral de Potencia ($\mu V^2 / Hz$)', fontsize=12)
plt.title('Comparación Espectral Corregida: Baseline vs. Ojos Cerrados', fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=11, loc='upper right')
plt.tight_layout()

# Guardar la imagen
plt.savefig('comparativa_alpha_zoom.png', dpi=300)
plt.show()