import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# 1. Cargar el archivo de Ojos Cerrados (usando la ruta que tengas)
signals = pd.read_csv('ojoscerrados/ojoscerrados.csv', delimiter=',')

# 2. Recortar los primeros 3 segundos (750 filas) para limpiar el inicio
df_limpio = signals.iloc[4500:].reset_index(drop=True)
data = df_limpio.values

# Tomamos un canal posterior/occipital si es posible (ch_7 o ch_8 suelen serlo)
# Por ahora usemos ch_1 (índice 2) o ch_7 (índice 8) para comparar. Probemos con índice 2:
eeg = data[:, 2] 
eeg = eeg - np.mean(eeg)
sfreq = 250.0  # 250 Hz

# 3. Calcular el Espectro de Potencia (PSD) con el método de Welch
# Usamos nperseg=1024 para tener muy buena definición entre 8 y 12 Hz
frecuencias, potencia = welch(eeg, fs=sfreq, nperseg=1024)

# 4. Encontrar automáticamente la frecuencia pico dentro del rango Alpha (8-12 Hz)
indices_alpha = np.where((frecuencias >= 8) & (frecuencias <= 12))[0]
frecuencias_alpha = frecuencias[indices_alpha]
potencia_alpha = potencia[indices_alpha]

# Buscamos el máximo valor de potencia en ese rango
indice_max_alpha = np.argmax(potencia_alpha)
frecuencia_pico = frecuencias_alpha[indice_max_alpha]
potencia_pico = potencia_alpha[indice_max_alpha]

print("-" * 50)
print(f"¡ANÁLISIS DE RITMO ALPHA COMPLETO!")
print(f"Frecuencia dominante encontrada en rango Alpha: {frecuencia_pico:.2f} Hz")
print(f"Potencia del pico: {potencia_pico:.4f} uV^2/Hz")
print("-" * 50)

# 5. Graficar el espectro enfocado
plt.figure(figsize=(10, 5))
plt.plot(frecuencias, potencia, color='darkblue', linewidth=2, label='Señal EEG (Ojos Cerrados)')

# Resaltar el área Alpha (8-12 Hz) con un sombreado
plt.axvspan(8, 12, color='orange', alpha=0.2, label='Banda Alpha (8-12 Hz)')

# Marcar el pico encontrado
plt.plot(frecuencia_pico, potencia_pico, 'ro', markersize=8, label=f'Pico Alpha a {frecuencia_pico:.2f} Hz')

# Ajustamos los límites de visualización (0 a 35 Hz es ideal para ver ritmos cerebrales)
plt.xlim([1, 35])
plt.ylim([0, max(potencia[(frecuencias >= 1) & (frecuencias <= 35)]) * 1.1])

plt.xlabel('Frecuencia (Hz)', fontsize=11)
plt.ylabel('Densidad Espectral de Potencia ($\mu V^2 / Hz$)', fontsize=11)
plt.title('Detección de Ritmo Alpha - Condición: Ojos Cerrados', fontsize=13, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()

# Guardar para el PDF final
plt.savefig('deteccion_alpha_ojos_cerrados.png', dpi=300)
plt.show()