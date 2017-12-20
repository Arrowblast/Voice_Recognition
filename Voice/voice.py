import matplotlib.pyplot as plt
import soundfile as sf # get the api
import glob
from scipy.fftpack import fft
import numpy as np
from pylab import *
from scipy.signal import decimate, blackmanharris,hann,stft

def parabolic(f, x):
    xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)
def hps(data, fs,range):

    n=len(data)
    if(type(data[0]) is not np.float64):
        data=[d[0] for d in data]
    m=mean(data)
    data=[d-m for d in data]

    signal=data

    windowed=signal*hann(n)

    X = abs(fft(windowed))
    best=0
    hps = copy(X)
    for h in arange(2, range): # TODO: choose a smarter upper limit
        dec = decimate(X, h)
        hps[:len(dec)] += dec




    i_peak = argmax(hps[:len(dec)])
    i_interp=parabolic(hps[:len(dec)],i_peak)[0]
    wynik=fs * i_interp / n
    print(wynik)
    return wynik

files=glob.glob("train/train/*.wav")
ile=0
for f in files:
    data, fs = sf.read(f)
    real_value=f.split('_')[1].split(".")[0]
    range=5
    wynik=1000
    roz="unknown"
    for range in np.arange(5,7):
        wynik = hps(data, fs,range)
        if(wynik>85 and wynik<180):
            roz="M"
            break
        elif(wynik>165 and wynik<255):
            roz="K"
            break;
        else:
            roz="unknown"

    print(real_value+" "+roz)
    if(real_value==roz):
        ile+=1
print(ile)

