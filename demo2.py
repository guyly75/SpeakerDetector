from resemblyzer import preprocess_wav, VoiceEncoder
from demo_utils import *
from pathlib import Path
from scipy.io.wavfile import write
from datetime import datetime
import os

def is_speaking(similarity_value):
    return similarity_value>=0.80 if True else False


def speaker_chunks(wav,wav_splits,speaker_values):
    seconds_per_window=wav_splits[0].stop/16000
    is_speaker_speaking=is_speaking(speaker_values[0])
    start_speak=0
    for position,value in enumerate(speaker_values):
        if is_speaker_speaking == is_speaking(value): # no change
            continue
        elif is_speaker_speaking: # stop speaking
            if is_speaking(max(speaker_values[position:int(position+(seconds_per_window*10))])):
                continue
            else:
                yield wav_splits[start_speak].start,wav_splits[position-1].stop
        else: # starts to speak
            start_speak=position
        is_speaker_speaking=is_speaking(value)
    return


print(f'starting: {datetime.now()}')

## Get reference audios
# Load the interview audio from disk
# Source for the interview: https://www.youtube.com/watch?v=X2zqiX6yL3I

input_filename=r'H:\My Drive\Speakers Project\Data\Audio Extraction\1293ID-Session-547662.wav'
wav_fpath = Path(input_filename)
#wav_fpath = Path(r'X2zqiX6yL3I.mp3')
rubi_fpath= Path(r'H:\My Drive\Speakers Project\Data\Audio Extraction\speakers\rubi.wav')
merav_fpath= Path(r'H:\My Drive\Speakers Project\Data\Audio Extraction\speakers\MeiravMichaeli.wav')
wav = preprocess_wav(wav_fpath)
rubi_wav=preprocess_wav(rubi_fpath)
merav_wav=preprocess_wav(merav_fpath)

speaker_names = ["Rubi Rivlin","Merav Michaeli"]
speaker_wavs = [rubi_wav,merav_wav]
    
## Compare speaker embeds to the continuous embedding of the interview
# Derive a continuous embedding of the interview. We put a rate of 16, meaning that an 
# embedding is generated every 0.0625 seconds. It is good to have a higher rate for speaker 
# diarization, but it is not so useful for when you only need a summary embedding of the 
# entire utterance. A rate of 2 would have been enough, but 16 is nice for the sake of  the 
# demonstration. 
# We'll exceptionally force to run this on CPU, because it uses a lot of RAM and most GPUs 
# won't have enough. There's a speed drawback, but it remains reasonable.
encoder = VoiceEncoder("cpu")
print("Running the continuous embedding on cpu, this might take a while...")
_, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True, rate=3)


# Get the continuous similarity for every speaker. It amounts to a dot product between the 
# embedding of the speaker and the continuous embedding of the interview
speaker_embeds = [encoder.embed_utterance(speaker_wav) for speaker_wav in speaker_wavs]
similarity_dict = {name: cont_embeds @ speaker_embed for name, speaker_embed in 
                   zip(speaker_names, speaker_embeds)}
print(f'similaroty dictionary done: {datetime.now()}')        
for speaker in speaker_names:
    for counter,(start,stop) in enumerate(speaker_chunks(wav,wav_splits,similarity_dict[speaker])):
        write(f'output\\{os.path.basename(input_filename)} - {speaker}-{start/16000:.1f}-{stop/16000:.1f}.wav',16000,np.array(wav[start:stop]))

print(f'done: {datetime.now()}')        
