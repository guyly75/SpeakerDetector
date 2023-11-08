from resemblyzer import preprocess_wav, VoiceEncoder
from demo_utils import *
from pathlib import Path
from scipy.io.wavfile import write
from datetime import datetime,timedelta
import os 
import pickle
import librosa
import math


speaking_level=0.84

def is_speaking(similarity_value):
    return similarity_value>=speaking_level if True else False

def is_speaking_low(similarity_value):
    return similarity_value>=(speaking_level*0.95) if True else False

def new_picks(wav,wav_splits,speaker_values):
    window_size=10
    good_match={a:b for a,b in enumerate(speaker_values) if b>speaking_level}
    last_position=0
    for _,ind in enumerate(good_match):
        if ind<=last_position: 
            continue 
        if np.mean(speaker_values[ind-window_size:ind+window_size])>=speaking_level*0.95:
            for ind1 in range(ind+1,len(speaker_values)-window_size):
                if np.mean(speaker_values[ind1-window_size:ind1+window_size])<0.7:
                    yield wav_splits[ind].start,wav_splits[ind1].stop
                    last_position=ind1
                    break
            #yield wav_splits[ind].start,wav_splits[-1].stop
    return

def speaker_chunks(wav,wav_splits,speaker_values):
    seconds_per_window=wav_splits[0].stop/16000
    is_speaker_speaking=is_speaking(speaker_values[0])
    start_speak=0
    for position,value in enumerate(speaker_values):
        if is_speaker_speaking == is_speaking(value): # no change
            continue
        elif is_speaker_speaking: # stop speaking
            if is_speaking_low(max(speaker_values[position:int(position+(seconds_per_window*100))])):
                continue
            else:
                yield wav_splits[start_speak].start,wav_splits[position-1].stop
        elif is_speaking_low(max(speaker_values[position:int(position+(seconds_per_window*10))])): # starts to speak
            start_speak=position
        is_speaker_speaking=is_speaking(value)
    return

def get_from_or_create_file(file_name,function_to_execute):
    ## Read or Generate a file using the function_to_execute
    my_file = Path(file_name)
    if my_file.exists():
        with open(my_file, "rb") as infile:
            try:            
                return pickle.load(infile)
            except:
                print("error in file")

    ret=function_to_execute()    
    with open(my_file, "wb") as outfile:
        pickle.dump(ret, outfile)
    return ret
    


def embed_uttorence_chunks(input_filename,return_partials,rate):        
    ## Compare speaker embeds to the continuous embedding of the interview
    # Derive a continuous embedding of the interview. We put a rate of 16, meaning that an 
    # embedding is generated every 0.0625 seconds. It is good to have a higher rate for speaker 
    # diarization, but it is not so useful for when you only need a summary embedding of the 
    # entire utterance. A rate of 2 would have been enough, but 16 is nice for the sake of  the 
    # demonstration. 
    # We'll exceptionally force to run this on CPU, because it uses a lot of RAM and most GPUs 
    # won't have enough. There's a speed drawback, but it remains reasonable.

    wav=get_from_or_create_file(input_filename+'XX.dat',lambda: preprocess_wav(Path(input_filename)))

    encoder = VoiceEncoder("cpu")

    folder=os.path.dirname(input_filename)
    file_name=os.path.basename(input_filename)
    files_list = list(Path(folder).glob(file_name+'-embed*'))
    if len(files_list)>0:
        checked_file=files_list[0].name
        num_of_chunks=int(checked_file[checked_file.find('-embed')+6:checked_file.find('-embed')+7])
    else:
        num_of_chunks=1
    chunk_number=0    
    while True:
        chunk_size=math.floor(len(wav)/num_of_chunks)        
        chunk_start=chunk_size*chunk_number
        chunk_end=chunk_start+chunk_size
        try:            
            ret_embed = get_from_or_create_file(input_filename+f'-embed{num_of_chunks}-{chunk_number}-rate{rate}.dat',\
            lambda: encoder.embed_utterance(wav[chunk_start:chunk_end], return_partials, rate))

            (embed,partial_embeds,wav_slices)=ret_embed
            yield wav[chunk_start:chunk_end],partial_embeds,wav_slices

        except RuntimeError:
            num_of_chunks+=1
            chunk_number=0
            continue
        chunk_number+=1
        if chunk_number==num_of_chunks:
            return


def get_speakers_data():
    # assign directory
    encoder = VoiceEncoder("cpu")
    speakers_folder = r'F:\Drive\Speakers Project\Data\Audio Extraction\speakers'
    files = Path(speakers_folder).glob('*.wav')
    speakers_data={}
    wav_data={}
    for file in files:
        is_preproccesed=False
        if file.name[-11:]=='-prepro.wav':
            continue
        preproccessed_filename=str(file.resolve())+'-prepro.dat'
        speaker_name=os.path.basename(preproccessed_filename)[5:-11]     
#        prepro_file=Path(preproccessed_filename)
#        if prepro_file.exists():
#                is_preproccesed=True
#                file=prepro_file
#                speaker_name=os.path.basename(file)[5:-11]
#        else:
#            speaker_name=os.path.basename(file)[5:-4]
        print(f"processing {file}")            
        if speaker_name not in wav_data.keys():
            speakers_data[speaker_name]=[]
#            if is_preproccesed:
            wav_data[speaker_name]=get_from_or_create_file(preproccessed_filename,lambda: preprocess_wav(file,trim_long_silences=True))
#            else:
#                wav_data[speaker_name]=preprocess_wav(file)
#                write(preproccessed_filename,16000,wav_data[speaker_name])
        else:
            s=preprocess_wav(file,trim_long_silences=True)
            print(f"bfore",len(s),type(s))
            wav_data[speaker_name]+=s
            print(f"after")            
    print(f"generating embeds")
    for speaker_name in wav_data.keys():
        print(f"generating embeds for {speaker_name}")
        speakers_data[speaker_name]=get_from_or_create_file(f'{speakers_folder}\\{speaker_name}_utterance.dat',lambda: encoder.embed_utterance(wav_data[speaker_name]))
    return speakers_data


def process_file(input_filename,speakers_data):
    print(f'starting: {input_filename} - {datetime.now()}')
    prm_infile = Path(input_filename)
    if not prm_infile.exists():
        return
    

    ## Get reference audios
    # Load the interview audio from disk

    # dat_file=input_filename+'XX.dat';
    # wav_fpath = Path(input_filename)
    # my_file = Path(dat_file)
    # if my_file.exists():
    #     with open(dat_file, "rb") as infile:
    #         wav= pickle.load(infile)
    # else:
    #     wav=preprocess_wav(wav_fpath)    
    #     with open(input_filename+'XX.dat', "wb") as outfile:
    #         pickle.dump(wav, outfile);
    speaker_names = speakers_data.keys()
    speaker_embeds = speakers_data.values()
        
    print("Running the continuous embedding on cpu, this might take a while...")

    for part,(processed_wav, cont_embeds, wav_splits) in enumerate(embed_uttorence_chunks(input_filename,True,2)):

        # Get the continuous similarity for every speaker. It amounts to a dot product between the 
        # embedding of the speaker and the continuous embedding of the interview
        #speaker_embeds = [encoder.embed_utterance(speaker_wav) for speaker_wav in speaker_wavs]
        similarity_dict = {name: cont_embeds @ speaker_embed for name, speaker_embed in 
                        zip(speaker_names, speaker_embeds)}
        print(f'\n*********  S I M I L A R I T Y        V A L U E S for {os.path.basename(input_filename)}***************')
        for speaker in speaker_names:    
            print(f'{speaker[:-4]}: max similarity: {max(similarity_dict[speaker])}')
            if speaker[-7:]=="-prepro":
                speaker=speaker[:-7]    
            for counter,(start,stop) in enumerate(new_picks(processed_wav,wav_splits,similarity_dict[speaker])):
                folder=f'F:\\Drive\\Speakers Project\\processed\\{speaker}'
                if not Path(folder).exists():
                    os.mkdir(folder)
                max_simularity=round(max(similarity_dict[speaker]) * 100,1)
                mean_simularity=round(np.mean(similarity_dict[speaker]) * 100,1)
                write(f'{folder}\\{max_simularity}%-{mean_simularity}%-{os.path.basename(input_filename)}-{part}-{speaker}-{(datetime.min+timedelta(seconds=start/16000)):%H`%M`%S}-{(datetime.min+timedelta(seconds=stop/16000)):%H`%M`%S}.wav',16000,np.array(processed_wav[start:stop]))

        print(f'*********  D O N E: {os.path.basename(input_filename)}  ***************\n')                     
       
    print(f'complete: {input_filename} - {datetime.now()}')     

#sessions_folder = r'F:\\Drive\Speakers Project\Data\Audio Extraction'
#files = Path(sessions_folder).glob('*.wav')
#for file in files:
    #process_file(file)

def main():
    speakers_data=get_speakers_data()

    sessions_folder = r'F:\Drive\Speakers Project\Data\Audio Extraction'
    files = Path(sessions_folder).glob('*.wav')
    for file in files:        
        # if not file.name.startswith("1901"):
        #     continue
        prm_infile = Path(str(file)+'XX.dat')
        #if not prm_infile.exists():
        try: 
            process_file(str(file),speakers_data)
        except:
            print("exception raised in file: "+str(file))

    print(f'done: {datetime.now()}')        

main()