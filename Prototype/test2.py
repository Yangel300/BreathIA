import time
import math
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

# I2C setup
i2c = busio.I2C(board.SCL, board.SDA)

ads = ADS.ADS1115(i2c)
ads.gain = 8
ads.data_rate = 860

mic = AnalogIn(ads, 0)

SAMPLES = 60
DURATION = 20  # seconds
VREF = 1.0  # referencia para dB relativos

# Open file
with open("output.txt", "w") as f:
    start_time = time.time()

    while (time.time() - start_time) < DURATION:
        values = []

        # collect samples
        for _ in range(SAMPLES):
            v = mic.voltage
            values.append(v)

        # compute mean
        mean = sum(values) / SAMPLES

        # compute RMS
        squared = [(v - mean) ** 2 for v in values]
        rms = math.sqrt(sum(squared) / SAMPLES)

        # convert to dB (relative)
        if rms > 0:
            dB = 20 * math.log10(rms / VREF)
        else:
            dB = -100  # valor bajo si no hay señal

        # Save to file
        f.write(f"{dB}\n")

        # Optional: also print
        print(dB)

print("Recording finished. Data saved to output.txt")
