import time
from itertools import chain

import h5py
import glob
import numpy as np
import librosa
from re import sub
from loguru import logger
from pathlib import Path
from tqdm import tqdm
from tools.file_io import load_csv_file, write_pickle_file


def eval_metadata(dataset,csv_file):
    csv_list = load_csv_file(csv_path)
    captions=[]
    for i, item in enumerate(csv_list):
        item_captions = _sentence_process(item['caption'])
        captions.append(item_captions)
    eval_meta_dict = {'captions': np.array(captions)}
    
    return eval_meta_dict



def pack_dataset_to_hdf5(dataset):
    
    #dataset= 'Clotho_new'
    split = 'eval'

    sampling_rate = 32000
    all_captions = []

    if dataset == 'Clotho_new':  #새로운 eval set으로 h5 file 생성
        audio_duration = 30
    else:
        raise NotImplementedError(f'No dataset named: {dataset}')

    max_audio_length = audio_duration * sampling_rate

    csv_path = 'data/{}/csv_files/{}.csv'.format(dataset, split)
    audio_dir = 'data/{}/waveforms/{}/'.format(dataset, split)
    hdf5_path = 'data/{}/hdf5s/{}/'.format(dataset, split)

    # make dir for hdf5
    Path(hdf5_path).mkdir(parents=True, exist_ok=True)

    captions = []
    
    csv_list = load_csv_file(csv_path)
    captions=[]
    
    for i, item in enumerate(csv_list):
        item_captions = _sentence_process(item['caption'])
        captions.append(item_captions)
    eval_meta_dict = {'captions': np.array(captions)}
    
    audio_nums = len(meta_dict['captions'])
    
    a = glob.glob('/home/user/audio-text_retrieval/data/Clotho_new/waveforms/eval/*wav')
    file_names=[]
    for i in a:
        x= os.path.basename(i)
        file_names.append(x)

        
    start_time = time.time()

    with h5py.File(hdf5_path+'{}.h5'.format(split), 'w') as hf:

        hf.create_dataset('audio_name', shape=(audio_nums,), dtype=h5py.special_dtype(vlen=str))
        hf.create_dataset('audio_length', shape=(audio_nums,), dtype=np.uint32)
        hf.create_dataset('waveform', shape=(audio_nums, max_audio_length), dtype=np.float32)
        hf.create_dataset('caption', shape=(audio_nums, ), dtype=h5py.special_dtype(vlen=str))
        
        #for i, file_name in enumerate(file_names):
        for i, file_name in enumerate(tqdm(file_names)):
            wav_file_path = os.path.join(audio_dir, file_name)
            audio_name = file_name
            audio,_ = librosa.load(wav_file_path, sr=sampling_rate, mono=True) 
            audio, audio_length = pad_or_truncate(audio, max_audio_length)
            hf['audio_name'][i] = audio_name.encode()
            hf['audio_length'][i] = audio_length
            hf['waveform'][i] = audio
            hf['caption'][i] = meta_dict['captions'][i]

        logger.info(f'Packed {split} set to {hdf5_path} using {time.time() - start_time} s.')
    # words_list, words_freq = _create_vocabulary(all_captions)
    # logger.info(f'Creating vocabulary: {len(words_list)} tokens!')
    # write_pickle_file(words_list, 'data/{}/pickles/words_list.p'.format(dataset))
    

def _create_vocabulary(captions):
    vocabulary = []
    for caption in captions:
        caption_words = caption.strip().split()
        vocabulary.extend(caption_words)
    words_list = list(set(vocabulary))
    words_list.sort(key=vocabulary.index)
    words_freq = [vocabulary.count(word) for word in words_list]
    words_list.append('<sos>')
    words_list.append('<eos>')
    words_list.append('<ukn>')
    words_freq.append(len(captions))
    words_freq.append(len(captions))
    words_freq.append(0)

    return words_list, words_freq


def _sentence_process(sentence, add_specials=False):

    # transform to lower case
    sentence = sentence.lower()

    if add_specials:
        sentence = '<sos> {} <eos>'.format(sentence)

    # remove any forgotten space before punctuation and double space
    sentence = sub(r'\s([,.!?;:"](?:\s|$))', r'\1', sentence).replace('  ', ' ')

    # remove punctuations
    sentence = sub('[,.!?;:\"]', ' ', sentence).replace('  ', ' ')
    return sentence


def pad_or_truncate(x, audio_length):
    """Pad all audio to specific length."""
    length = len(x)
    if length <= audio_length:
        return np.concatenate((x, np.zeros(audio_length - length)), axis=0), length
    else:
        return x[:audio_length], audio_length