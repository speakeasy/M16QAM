import numpy as np
from random import randint
from commpy.modulation import QAMModem
from commpy.filters import rrcosfilter
from commpy.utilities import bitarray2dec, dec2bitarray

np.set_printoptions(threshold=np.nan)
N = 1024  # output size
M = 16
mod1 = QAMModem(M)  # QAM16
sB = dec2bitarray(randint(0, 2**(mod1.num_bits_symbol*N*M/4)), (mod1.num_bits_symbol*N*M/4))  # Random bit stream
# print np.array2string(sB)
sQ = mod1.modulate(sB)  # Modulated baud points
print np.array2string(np.abs(sQ))
sPSF = rrcosfilter(N*4, 0.2, 1, 4000)[1]
qW = np.convolve(sPSF, sQ) # Waveform with PSF
#print np.array2string(qW)