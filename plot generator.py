import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt,resample

def bandpass_filter(signal, fs, lowcut=0.5, highcut=40, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)
    return filtered

# Download a sample ECG record (e.g., record "100" from MIT-BIH Arrhythmia Database)
record = wfdb.rdrecord('100', pn_dir='mitdb', channels=[0])  # Lead I
ecg = record.p_signal.flatten()
ecg = ecg*2/1000
fs = record.fs
t = [i / fs for i in range(len(ecg))]
duration = 5 

filtered_ecg = bandpass_filter(ecg, fs)

fs_target = 1000
num_samples = int(len(filtered_ecg) * fs_target / fs)
filtered_resampled = resample(filtered_ecg, num_samples)
t_resampled = np.linspace(0, duration, num_samples)

gain = 137.8
bias = 0.4
amplified = filtered_ecg * gain +bias

def adc_simulate(signal, bits=6, vref=1.2):
    signal_clipped = np.clip(signal, 0, vref)
    levels = 2 ** bits
    step = vref / levels
    digital = np.round(signal_clipped / step)
    return digital, step

adc_bits = 6
adc_output, lsb = adc_simulate(amplified, bits=adc_bits, vref=1.2)
reconstructed = adc_output * lsb


plt.figure(figsize=(14, 8))

plt.subplot(3, 1, 1)
plt.plot(t[:2000], ecg[:2000], label='Noisy ECG (µV)')
plt.title('Noisy Input ECG')
plt.ylabel('Amplitude (µV)')
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(t[:2000], filtered_ecg[:2000], label='Filtered ECG (µV)', color='orange')
plt.title('Filtered ECG')
plt.ylabel('Amplitude (µV)')
plt.grid()

plt.subplot(3, 1, 3)
plt.step(t[:2000], reconstructed[:2000], where='mid', label='ADC Output (V)', color='green')
plt.title('Quantized Output (Post-ADC)')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.grid()
plt.tight_layout()
plt.show()

