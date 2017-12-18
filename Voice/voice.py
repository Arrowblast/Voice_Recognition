import matplotlib.pyplot as plt
import soundfile as sf # get the api
import glob
from scipy.fftpack import fft
import numpy as np
from pylab import *
from scipy.signal import decimate, blackmanharris

def parabolic(f, x):
    xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)
def hps(filename):
    data, fs = sf.read(filename) # load the data
    if(type(data[0]) is not np.float64):
        data=[mean(d) for d in data]
    n=len(data)
    signal=data

    windowed=signal*blackmanharris(n)

    X = log(abs(rfft(windowed)))

    hps = copy(X)
    for h in arange(2, 5): # TODO: choose a smarter upper limit
        dec = decimate(X, h)
        hps[:len(dec)] += dec
    i_peak = argmax(hps[:len(dec)])
    i_interp=parabolic(hps[:len(dec)],i_peak)[0]

    return fs * i_interp / n

files=glob.glob("train/train/*.wav")
ile=0
for f in files:
    real_value=f.split('_')[1].split(".")[0]

    wynik = (hps(f))
    if(wynik>85 and wynik<180):
        wynik="M"
    elif(wynik>165 and wynik<255):
        wynik="K"
    else:
        wynik="unknown"
    print(real_value+" "+wynik)
    if(real_value==wynik):
        ile+=1
print(ile)

