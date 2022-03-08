from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

frecuencia_muestreo = 44100
frecuencia = 440
tiempos = np.linspace(0.0, 1.0, frecuencia_muestreo)
amplitud = np.iinfo(np.int16).max

ciclos = frecuencia * tiempos

fracciones, enteros = np.modf(ciclos)
data = fracciones

data = fracciones - 0.5

data = np.abs(data)

data = data - data.mean()

alto, bajo = abs(max(data)), abs(min(data))
data = amplitud * data / max(alto, bajo)

fig, ejes = plt.subplots(2,4)

# plt.figure()
ejes[0,0].plot(tiempos, data)
# plt.show()

write("triangular.wav", frecuencia_muestreo, data.astype(np.int16))

cantidad_muestras = len(data)
periodo_muestreo = 1.0 / frecuencia_muestreo
transformada = np.fft.rfft(data)
frecuencias = np.fft.rfftfreq(cantidad_muestras, periodo_muestreo)

# plt.figure()
ejes[0,1].plot(frecuencias, np.abs(transformada))
# plt.show()

# 1.- Obtener en Hz las frecuencias de los armoicos de la señal

armonicos = frecuencias[transformada > 150000]
print(armonicos)

# 2.- Aplicarle un filtro pasa bajas que solo deje pasar la Freq Fundamental 
# y luego aplicarle la transformada inversa
# graficarla en dominio del tiempo
# y crear un archivo wav para escucharla

pasa_bajas = transformada.copy()
pasa_bajas[frecuencias > (frecuencia + 5)] *= 0

# plt.figure()
ejes[0,3].plot(frecuencias, np.abs(pasa_bajas), label = "Pasa bajas")
ejes[0,3].legend()
# plt.show

pasa_bajas_data = np.fft.irfft(pasa_bajas)
# plt.figure()
ejes[0,2].plot(tiempos, pasa_bajas_data)

write("pasa_bajas.wav", frecuencia_muestreo, pasa_bajas_data.astype(np.int16))

ciclos_rectangular = frecuencia * tiempos
fracciones_rectangular, enteros_rectangular = np.modf(ciclos_rectangular)
data_rectangular = amplitud * np.sign(fracciones_rectangular - fracciones_rectangular.mean())

ejes[1,0].plot(tiempos, data_rectangular)

transformada_rectangular = np.fft.rfft(data_rectangular)
frecuencias_rectangular = np.fft.rfftfreq(cantidad_muestras, periodo_muestreo)

ejes[1,1].plot(frecuencias_rectangular, np.abs(transformada_rectangular))

pasa_bajas_rectangular = transformada_rectangular.copy()
pasa_bajas_rectangular[frecuencias_rectangular > (frecuencia + 5)] *= 0

ejes[1,3].plot(frecuencias_rectangular, np.abs(pasa_bajas_rectangular))

pasa_bajas_data_rectangular = np.fft.irfft(pasa_bajas_rectangular)

ejes[1,2].plot(tiempos, pasa_bajas_data_rectangular)

plt.show()
