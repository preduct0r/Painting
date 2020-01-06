from scipy.io import wavfile


sr, wav_data = wavfile.read(f_name)
start = 1
end = 3

kusok_wavki = wav_data[int(start* sr): int(end * sr)]

