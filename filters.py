import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
from scipy.signal import butter, sosfiltfilt, sosfreqz

class filters:

    fs = None
    lowcut = None
    highcut = None
    order = None

    def __init__(self, fs=250000, lowcut=18500, highcut=37000, order=12):
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order

    def butter_bandpass(self, lowcut, highcut, fs, order=12, setorder=False):
        if setorder == False:
            order = self.order
        nyq = 0.5 * fs
        low = self.lowcut / nyq
        high = self.highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos

    def butter_bandpass_filter(self, data, lowcut, highcut):
        sos = self.butter_bandpass(lowcut, highcut, self.fs)
        y = sosfiltfilt(sos, data)
        return y

    def plot_response(self):
        plt.figure(1)
        plt.clf()
        for order in [3, 6, 9, 12]:
            sos = self.butter_bandpass(self.lowcut, self.highcut, self.fs, order=order)
            w, h = sosfreqz(sos, worN=2000)
            plt.plot((self.fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

        plt.plot([0, 0.5 * self.fs], [np.sqrt(0.5), np.sqrt(0.5)],
                 '--', label='sqrt(0.5)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain')
        plt.grid(True)
        plt.legend(loc='best')
        plt.show()

#flt = filters(250000, 18500, 23000, 12)
#flt.plot_response()