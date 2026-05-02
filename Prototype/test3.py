import time
import math
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
import numpy as np

# I2C setup
i2c = busio.I2C(board.SCL, board.SDA)

ads = ADS.ADS1115(i2c)
ads.gain = 8
ads.data_rate = 860  # máximo real

mic = AnalogIn(ads, 0)

TARGET_FS = 22050
REAL_FS = 860

SAMPLES = 60
DURATION = 20
VREF = 1.0

# factor de interpolación
UPSAMPLE_FACTOR = int(TARGET_FS / REAL_FS)

with open("output.txt", "w") as f:
    start_time = time.time()

    while (time.time() - start_time) < DURATION:
        values = []

        for _ in range(SAMPLES):
            v = mic.voltage
            values.append(v)

        # interpolación simple
        x = np.arange(len(values))
        x_new = np.linspace(0, len(values)-1, len(values)*UPSAMPLE_FACTOR)
        values_interp = np.interp(x_new, x, values)

        # RMS
        mean = np.mean(values_interp)
        rms = np.sqrt(np.mean((values_interp - mean) ** 2))

        if rms > 0:
            dB = 20 * math.log10(rms / VREF)
        else:
            dB = -100

        f.write(f"{dB}\n")
        print(dB)

print("Recording finished. Data saved to output.txt")
