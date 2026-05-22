#betina script EEG
print(__doc__)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import StringIO

signals = pd.read_csv('blinking/blinking.csv', delimiter=',')

print('Estructura de la informacion:')
print(signals.head())

data = signals.values
eeg = data[:, 2] # Ch_1 

# Descartamos los primeros 2 segundos (500 muestras) para evitar el pico de acople inicial
eeg = eeg[2500:]

###############################################################################################
# La operación de convolución permite implementar el suavizado del Moving Average
windowlength = 10
avgeeg = np.convolve(eeg, np.ones((windowlength,))/windowlength, mode='same')

# El kernel/máscara está compuesto de 10 valores de 1/10.  Cuando esos valores se suman para cada posición, implica que se reemplaza el valor por el promedio
# de los 5 valores anteriores y 4 posteriores.

plt.plot(avgeeg,'r', label='EEG')
plt.xlabel('t');
plt.ylabel('eeg(t)');
plt.title(r'Smoothed EEG Signal')     
plt.ylim([-200, 200]);
plt.xlim([0,len(avgeeg)])
#plt.savefig('images/smoothed.png')
plt.show()


# Scaling and normalizing are somehow temporal filters

# Feature scaling, Xnew = Xold / Xmax, everything will be on the range 0-1

def simple_feature_scaling(arr):
    """This method applies simple-feature-scaling
        to a distribution (arr).
    @param arr: An array or list or series object
    @return: The arr with all features simply scaled
    """

    arr_max = max(arr)
    new_arr = [i/arr_max for i in arr]

    return new_arr
  
# Let's define an array arr
  
eeg_scaled = simple_feature_scaling(eeg)

print(f'Before Scaling...\n min  is {min(eeg)}\n max  is {max(eeg)}\n')
print(f'After Scaling...\n min is {min(eeg_scaled)}\n max is {max(eeg_scaled)}')

plt.plot(eeg_scaled,'r', label='EEG')
plt.xlabel('t');
plt.ylabel('eeg(t)');
plt.title(r'Scaled EEG Signal')     
plt.ylim([-2, 2]);
plt.xlim([0,len(avgeeg)])
#plt.savefig('images/scaledeeg.png')
plt.show()

# Min-Max Scaling, Xnew = (Xold - Xmin) / (Xmax - Xmin)

def min_max_scaling(arr):
    """This method applies min-max-scaling
        to a distribution (arr).
    @param arr: An array or list or series object
    @return: The arr with all features min-max scaled
    """

    arr_max = max(arr)
    arr_min = min(arr)
    range_ = arr_max - arr_min

    new_arr = [(i-arr_min)/range_ for i in arr]

    return new_arr
  
  # Let's define an arr and call the min-max scaler
  
eeg_minmax = min_max_scaling(eeg)

print(f'Before Scaling...\n min  is {min(eeg)}\n max  is {max(eeg)}\n')
print(f'After Scaling...\n min is {min(eeg_minmax)}\n max is {max(eeg_minmax)}')

plt.plot(eeg_minmax,'r', label='EEG')
plt.xlabel('t');
plt.ylabel('eeg(t)');
plt.title(r'Min Max EEG Signal')     
plt.ylim([-2, 2]);
plt.xlim([0,len(avgeeg)])
#plt.savefig('images/minmaxeeg.png')
plt.show()

# Normalization, "Statistical Normalization" means pushing the data to match to a Normal Distribution.
# (Normalization is also considered in terms of vector normalization, divide by the norm.)
# This is also call, "Standarization" = "Statistical Normalization"

# Z-Score, Xnew = Xold - Xmean / Xstd, everything is around -1 to 1

def z_score_norm(arr):
    """Apply z-score normalization
        to an array or series
    """
    mean_ = np.mean(arr)
    std_ = np.std(arr)

    new_arr = [(i-mean_)/std_ for i in arr]

    return new_arr

eeg_zscore = z_score_norm(eeg)

print(f'Before ZScore...\n min  is {min(eeg)}\n max is {max(eeg)}\n')
print(f'After ZScore...\n min is {min(eeg_zscore)}\n max  is {max(eeg_zscore)}')

plt.plot(eeg_zscore,'r', label='EEG')
plt.xlabel('t');
plt.ylabel('eeg(t)');
plt.title(r'z-score EEG Signal')     
plt.ylim([-20, 20]);
plt.xlim([0,len(avgeeg)])
#plt.savefig('images/zscoredeeg.png')
plt.show()


# Box-Cox Normalization:  Transform the data to look more normal N()
from scipy import stats

eeg2 = eeg + abs(min(eeg)) + 1
eeg_boxcox, _ = stats.boxcox(eeg2)

print(f'Before BoxCox...\n min  is {min(eeg)}\n max  is {max(eeg)}\n')
print(f'After BoxCox...\n min is {min(eeg_boxcox)}\n max  is {max(eeg_boxcox)}')

plt.plot(eeg_boxcox,'r', label='EEG')
plt.xlabel('t');
plt.ylabel('eeg(t)');
plt.title(r'BoxCox Scaling EEG Signal')     
plt.ylim([-200, 200]);
plt.xlim([0,len(avgeeg)])
#plt.savefig('images/boxcoxeeg.png')
plt.show()


# Get a gaussian kernel for smoothing.
mu = 0
sigma = 1
x = np.linspace(-4,4,1000)
gaussian = np.exp(-((x - mu)**2)/(2 * sigma**2) / ( np.sqrt(2 * np.pi) * sigma))

gaussian_kernel = gaussian[[1,250,500,750,999]]