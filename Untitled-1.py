from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np

fpath = Path(r'H:\My Drive\Speakers Project\Data\Audio Extraction\1256ID-Session-473408.wav')
wav = preprocess_wav(fpath)


encoder = VoiceEncoder()
embed = encoder.embed_utterance(wav,return_partials=True)
np.set_printoptions(precision=3, suppress=True)
print(embed)