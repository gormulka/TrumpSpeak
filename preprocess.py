import glob
from random import Random

from utils.display import *
from utils.dsp import *
from utils import hparams as hp
from multiprocessing import Pool, cpu_count
from utils.paths import Paths
import pickle
import argparse

from utils.text import clean_text
from utils.text.recipes import ljspeech
from utils.files import get_files, pickle_binary
from pathlib import Path


# Helper functions for argument types
def valid_n_workers(num):
    n = int(num)
    if n < 1:
        raise argparse.ArgumentTypeError('%r must be an integer greater than 0' % num)
    return n

parser = argparse.ArgumentParser(description='Preprocessing for WaveRNN and Tacotron')
parser.add_argument('--path', '-p', help='directly point to dataset path (overrides hparams.wav_path')
parser.add_argument('--extension', '-e', metavar='EXT', default='.wav', help='file extension to search for in dataset folder')
parser.add_argument('--num_workers', '-w', metavar='N', type=valid_n_workers, default=cpu_count()-1, help='The number of worker threads to use for preprocessing')
parser.add_argument('--hp_file', metavar='FILE', default='hparams.py', help='The file to use for the hyperparameters')
args = parser.parse_args()

hp.configure(args.hp_file)  # Load hparams from file
if args.path is None:
    args.path = hp.wav_path

extension = args.extension
path = args.path


def convert_file(path: Path):
    y = load_wav(path)
    peak = np.abs(y).max()
    if hp.peak_norm or peak > 1.0:
        y /= peak
    mel = melspectrogram(y)
    if hp.voc_mode == 'RAW':
        quant = encode_mu_law(y, mu=2**hp.bits) if hp.mu_law else float_2_label(y, bits=hp.bits)
    elif hp.voc_mode == 'MOL':
        quant = float_2_label(y, bits=16)

    return mel.astype(np.float32), quant.astype(np.int64)


def process_wav(path: Path):
    wav_id = path.stem
    m, x = convert_file(path)
    np.save(paths.mel/f'{wav_id}.npy', m, allow_pickle=False)
    np.save(paths.quant/f'{wav_id}.npy', x, allow_pickle=False)
    text = text_dict[wav_id]
    text = clean_text(text)
    return wav_id, m.shape[-1], text


wav_files = get_files(path, extension)
paths = Paths(hp.data_path, hp.voc_model_id, hp.tts_model_id)

print(f'\n{len(wav_files)} {extension[1:]} files found in "{path}"\n')

if len(wav_files) == 0:

    print('Please point wav_path in hparams.py to your dataset,')
    print('or use the --path option.\n')

else:
    text_dict = ljspeech(path)

    n_workers = max(1, args.num_workers)

    simple_table([
        ('Sample Rate', hp.sample_rate),
        ('Bit Depth', hp.bits),
        ('Mu Law', hp.mu_law),
        ('Hop Length', hp.hop_length),
        ('CPU Usage', f'{n_workers}/{cpu_count()}'),
        ('Num Validation', hp.n_val)
    ])

    pool = Pool(processes=n_workers)
    dataset = []
    cleaned_texts = []
    for i, (item_id, length, cleaned_text) in enumerate(pool.imap_unordered(process_wav, wav_files), 1):
        if item_id in text_dict:
            dataset += [(item_id, length)]
            cleaned_texts += [(item_id, cleaned_text)]
        bar = progbar(i, len(wav_files))
        message = f'{bar} {i}/{len(wav_files)} '
        stream(message)

    random = Random(hp.seed)
    random.shuffle(dataset)
    train_dataset = dataset[hp.n_val:]
    val_dataset = dataset[:hp.n_val]
    # sort val dataset longest to shortest
    val_dataset.sort(key=lambda d: -d[1])

    for id, text in cleaned_texts:
        text_dict[id] = text

    pickle_binary(text_dict, paths.data/'text_dict.pkl')
    pickle_binary(train_dataset, paths.data/'train_dataset.pkl')
    pickle_binary(val_dataset, paths.data/'val_dataset.pkl')

    print('\n\nCompleted. Ready to run "python train_tacotron.py" or "python train_wavernn.py". \n')
