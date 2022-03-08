from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

frecuencia_muestreo = 44100.0
frecuencia = 440.0
duracion = (1 / frecuencia) * 4.58
tiempos = np.linspace(0.0, duracion, int(duracion * frecuencia_muestreo))
amplitud = np.iinfo(np.int16).max

# f(t) = A sin (2 * pi * f * t)
data = amplitud * np.sin(2.0 * np.pi * frecuencia * tiempos)

fig, ejes = plt.subplots(1,2)

# plt.figure()
ejes[0].plot(tiempos, data)
# plt.show()

cantidad_muestras = len(data)
periodo_muestreo = 1.0 / frecuencia_muestreo
transformada = np.fft.rfft(data)
frecuencias = np.fft.rfftfreq(cantidad_muestras, periodo_muestreo)

# plt.figure()
ejes[1].plot(frecuencias, np.abs(transformada))

plt.show()
