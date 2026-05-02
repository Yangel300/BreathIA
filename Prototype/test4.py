import sounddevice as sd
import numpy as np
import math
import wave
import time

# Parámetros
FS = 22050          # sample rate real
DURATION = 20       # segundos
BLOCKSIZE = 1024    # muestras por bloque
VREF = 1.0

# buffers
audio_data = []
db_values = []

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)

    # mono (tomar canal 0)
    samples = indata[:, 0]

    # guardar audio
    audio_data.extend(samples)

    # RMS
    mean = np.mean(samples)
    rms = np.sqrt(np.mean((samples - mean) ** 2))

    # dB relativo
    if rms > 0:
        dB = 20 * math.log10(rms / VREF)
    else:
        dB = -100

    db_values.append(dB)
    print(dB)

# grabación
print("Recording...")
with sd.InputStream(samplerate=FS, channels=1, callback=audio_callback, blocksize=BLOCKSIZE):
    sd.sleep(DURATION * 1000)

print("Recording finished")

# guardar WAV
audio_np = np.array(audio_data)
audio_np = np.int16(audio_np / np.max(np.abs(audio_np)) * 32767)

with wave.open("output.wav", "w") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)  # 16 bits
    wf.setframerate(FS)
    wf.writeframes(audio_np.tobytes())

# guardar dB
with open("output_db.txt", "w") as f:
    for d in db_values:
        f.write(f"{d}\n")

print("Saved: output.wav + output_db.txt")
