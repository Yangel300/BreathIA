import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

i2c = busio.I2C(board.SCL, board.SDA)

ads = ADS.ADS1115(i2c)
ads.gain = 8
ads.data_rate = 860

mic = AnalogIn(ads, 0)

SAMPLES = 60

while True:
    total = 0
    values = []

    # collect samples
    for _ in range(SAMPLES):
        v = mic.voltage
        values.append(v)
        total += v

    mean = total / SAMPLES

    max_dev = 0
    for v in values:
        d = abs(v - mean)
        if d > max_dev:
            max_dev = d

    print(max_dev)
