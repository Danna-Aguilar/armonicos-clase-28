from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.io.wavfile import write

frecuencia_muestreo = 44100
frecuencia = 250
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

plt.figure()
plt.plot(tiempos, data)
plt.show()

write("triangular.wav", frecuencia_muestreo, data.astype(np.int16))

cantidad_muestras = len(data)
periodo_muestreo = 1.0 / frecuencia_muestreo
transformada = np.fft.rfft(data)
frecuencias = np.fft.rfftfreq(cantidad_muestras, periodo_muestreo)

plt.figure()
plt.plot(frecuencias, np.abs(transformada))
plt.show()

# 1.- Obtener en Hz las frecuencias de los armoicos de la señal

umbral = 100000
print(frecuencias[transformada > umbral])
transformada[transformada > umbral]

# 2.- Aplicarle un filtro pasa bajas que solo deje pasar la Freq Fundamental 
# y luego aplicarle la transformada inversa
# graficarla en dominio del tiempo
# y crear un archivo wav para escucharla

frecuencia_muestreo, muestras = wavfile.read("triangular.wav")

print(frecuencia_muestreo)
print("Tipo: " + str(type(muestras)))
print("Dtype (bithdept): " + str(muestras.dtype))
print("shape = " + str(muestras.shape))

canales = 1
if len(muestras.shape) == 1:
    print("# Canales = 1")
else:
    print("# Canales = " + str(muestras.shape[1]))
    canales = muestras.shape[1]
duracion = muestras.shape[0] / frecuencia_muestreo
print("duracion:  " + "{:.2f}".format(duracion) + " segs")

figura, ejes = plt.subplots(2,2)
if canales == 1:
    ejes[0,0].plot(tiempos, muestras, label="Canal mono")
else:
    ejes[0,0].plot(tiempos, muestras[:, 0], label="Izquierdo")
    ejes[0,0].plot(tiempos, muestras[:, 1], label="Derecho")

ejes[0,0].legend()
ejes[0,0].set(xlabel = "Tiempo (s)", ylabel = "Amplitud")

if canales > 1:
    data = muestras[:,0]
else:
    data = muestras

pasa_bajas = transformada.copy()
pasa_bajas[frecuencias > frecuencia] *=0

ejes[1,1].plot(frecuencias, np.abs(pasa_bajas), 
    label = "Espectro filtrado, pasa bajas")
ejes[1,1].legend()
ejes[1,1].set(xlabel = "Frecuencia (Hz)", ylabel = "Amplitud")

pasa_bajas_data = np.fft.irfft(pasa_bajas)
ejes[1,0].plot(tiempos, pasa_bajas_data, label = "Audio con pasa bajas")
ejes[1,1].legend()
ejes[1,1].set(xlabel = "Tiempo (s)", ylabel = "Amplitud")

write("pasa_bajas.wav", frecuencia_muestreo, pasa_bajas_data.astype(np.int16))

plt.show()