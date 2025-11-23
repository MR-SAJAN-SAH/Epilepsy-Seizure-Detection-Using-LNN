import pyedflib
import numpy as np

file = pyedflib.EdfReader("chb01/chb01_01.edf")
n = file.signals_in_file

signals = np.zeros((n, file.getNSamples()[0]))

for i in range(n):
    signals[i, :] = file.readSignal(i)

file.close()

print(signals.shape)
