from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
import torch
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
from models import Generator
import time
h = None
device = 'cpu'
import matplotlib.pyplot as plt

import numpy as np
# Original Source: https://github.com/originalauthor/originalproject
# This file has been modified from the original work
def qmf(weight):
    c = (torch.flip(weight, [2]))

    for i in range(c.size(2)):
        if i % 2 == 0:
            c[:, :, i] = -c[:, :, i]
    print(weight)
    print(c)
    print('----')
    return c
def plot_filter(filter1,filter2):
    import matplotlib
    matplotlib.use('Qt5Agg')
    # fig=plt.figure()
    # ax=fig.add_subplot(1,1,1)
    # ax.set(xlim=[0,9],ylim=[-3,3])
    x=np.arange(len(filter1))
    # ax.plot(x,filter1)
    # ax.plot(x,filter2)
    plt.plot(x,filter1,x,filter2)
    plt.show()
    plt.close()

    return 1

def load_checkpoint(filepath, device):
    print(filepath)
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(a):
    generator = Generator(h).to(device)

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    filelist = os.listdir(a.input_wavs_dir)

    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()
    total_length=0
    total_time=0
    # import numpy as np
    # model_parameters = filter(lambda p: p.requires_grad, generator.parameters())
    # params = sum([np.prod(p.size()) for p in model_parameters])
    # print(params,'params')a
    samples=0
    with torch.no_grad():
        for i, filname in enumerate(filelist):
            wav, sr = load_wav(os.path.join(a.input_wavs_dir, filname))
            wav = wav / MAX_WAV_VALUE
            wav = torch.FloatTensor(wav).to(device)
            samples=samples+wav.size(0)
            x = get_mel(wav.unsqueeze(0))
            start=time.time()
            y_g_hat,weight = generator(x)
            end=time.time()
            total_time=total_time+(end-start)
            print(end-start,'time')
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')

            output_file = os.path.join(a.output_dir, os.path.splitext(filname)[0] + '_generated.wav')
            write(output_file, h.sampling_rate, audio)
            print(output_file)

        print(20744190/total_time,'RTF')
        # print(a.checkpoint_file.split('/')[2],'asdffff')
        set=[]
        # for i in weight:
        #     for j in i:
        #         set.append(np.array(j))
        #
        # np.save(a.checkpoint_file.split('/')[2],np.stack(set))
        #
        # for i in weight:
        #     f1=i[0][0]
        #     f2=qmf(i)[0][0]
        #     f1=np.pad(f1,(0,100))
        #     f2=np.pad(f2,(0,100))
        #     a=np.absolute(np.fft.fft(f1))
        #     b=np.absolute(np.fft.fft(f2))
        #     plot_filter(f1,f2)
        #     plot_filter(20*np.log10(a),20*np.log10(b))
        #
        #     print(sum(sum(sum(i**2))))
        #     print(a+b)


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wavs_dir', default='test')
    parser.add_argument('--output_dir', default='generated_files')
    parser.add_argument('--checkpoint_file',default='./cp_hifigan/96ch_res1/g_01200000')
    a = parser.parse_args()

    config_file = os.path.join( 'wavelet_config_v5.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(a)


if __name__ == '__main__':
    main()

