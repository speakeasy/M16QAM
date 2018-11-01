import os
import wave
import librosa as lr
import numpy as np
import scipy.fftpack
from filters import filters
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.nan)


class wavfile:
    cwd = os.getcwd()
    filename = None
    wfile = None
    data = None
    samplerate = None
    fftx = None
    fftT = 0.000004
    fftN = None
    fftY = None

    def __init__(self, filename='file.wav'):
        self.filename = self.cwd + "\\" + filename
        self.wfile = wave.open(self.filename, "r")

    def read_wav(self, offset=0.0, duration=1.0):
        self.data = lr.core.load(self.filename, sr=self.wfile.getframerate(), mono=False, offset=offset, duration=duration)[0]

    def fft_wav(self, offset=0.0, duration=1.0, channel=0):
        if(len(self.data[channel]) < 1):
            self.data = self.read_wav(0.0, 1.0)
        self.fftN = len(self.data[channel])
        self.fftT = (duration / (self.wfile.getframerate() * 1.0))
        self.fftY = scipy.fftpack.fft(self.data[channel])
        self.fftx = np.linspace(0.0, 1.0 / (2.0 * self.fftT), self.fftN // 2)

    def show_fft(self):
        if(len(self.fftx) < 1):
            self.fft_wav()
        plt.plot(self.fftx, 2.0 / self.fftN * np.abs(self.fftY[0:self.fftN // 2]))
        plt.grid()
        plt.show()




wf = wavfile()
wf.read_wav(5.0685, 0.003)
filt = filters(wf.wfile.getframerate(), 21500, 25800, order=12)

t = np.linspace(0, len(wf.data[0]), len(wf.data[1]))
wf.fft_wav()

wf.data[0] = filt.butter_bandpass_filter(wf.data[0], filt.lowcut, filt.highcut)
wf.data[1] = filt.butter_bandpass_filter(wf.data[1], filt.lowcut, filt.highcut)


Amplitude = 2* np.sqrt(wf.data[0]**2 + wf.data[1]**2)
plt.plot(t, Amplitude)
plt.show()
wf.fft_wav(channel=0)
wf.show_fft()
wf.fft_wav(channel=1)
wf.show_fft()



