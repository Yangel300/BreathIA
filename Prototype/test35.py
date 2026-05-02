import time
import math
import wave
import board
import busio
import numpy as np
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

# =========================
# CONFIGURACIÓN
# =========================
TARGET_FS = 22050      # Frecuencia final del WAV
DURATION = 20          # segundos
VREF = 1.0             # referencia para dB relativo

OUTPUT_WAV = "output_ads1115.wav"
OUTPUT_DB = "output_db.txt"

# =========================
# I2C + ADS1115
# =========================
i2c = busio.I2C(board.SCL, board.SDA)

ads = ADS.ADS1115(i2c)
ads.gain = 8
ads.data_rate = 860    # máximo del ADS1115
mic = AnalogIn(ads, 0)

# =========================
# CAPTURA DURANTE 20 SEGUNDOS
# =========================
samples = []
timestamps = []

print("Recording...")

start_time = time.time()

while (time.time() - start_time) < DURATION:
    now = time.time() - start_time
    voltage = mic.voltage

    timestamps.append(now)
    samples.append(voltage)

print("Recording finished")

# =========================
# CONVERTIR A NUMPY
# =========================
samples = np.array(samples, dtype=np.float32)
timestamps = np.array(timestamps, dtype=np.float32)

# quitar offset DC
samples_ac = samples - np.mean(samples)

# =========================
# INTERPOLAR A 22050 Hz PARA WAV
# =========================
num_target_samples = int(DURATION * TARGET_FS)
target_times = np.linspace(0, DURATION, num_target_samples, endpoint=False)

audio_interp = np.interp(target_times, timestamps, samples_ac)

# normalizar a int16
max_val = np.max(np.abs(audio_interp))

if max_val > 0:
    audio_int16 = np.int16(audio_interp / max_val * 32767)
else:
    audio_int16 = np.zeros_like(audio_interp, dtype=np.int16)

# =========================
# GUARDAR WAV
# =========================
with wave.open(OUTPUT_WAV, "w") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)       # 16 bits
    wf.setframerate(TARGET_FS)
    wf.writeframes(audio_int16.tobytes())

# =========================
# CALCULAR dB POR BLOQUES
# =========================
block_size = 1024
db_values = []

for i in range(0, len(audio_interp), block_size):
    block = audio_interp[i:i + block_size]

    if len(block) == 0:
        continue

    rms = np.sqrt(np.mean(block ** 2))

    if rms > 0:
        dB = 20 * math.log10(rms / VREF)
    else:
        dB = -100

    db_values.append(dB)

# =========================
# GUARDAR dB
# =========================
with open(OUTPUT_DB, "w") as f:
    for dB in db_values:
        f.write(f"{dB}\n")

print(f"Samples captured: {len(samples)}")
print(f"Saved: {OUTPUT_WAV} + {OUTPUT_DB}")
