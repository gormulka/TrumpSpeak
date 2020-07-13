import time
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F

from trainer.voc_trainer import VocTrainer
from utils.display import stream, simple_table
from utils.dataset import get_vocoder_datasets
from utils.distribution import discretized_mix_logistic_loss
from utils import hparams as hp
from models.fatchord_version import WaveRNN
from gen_wavernn import gen_testset
from utils.paths import Paths
import argparse
from utils import data_parallel_workaround
from utils.checkpoints import save_checkpoint, restore_checkpoint


if __name__ == '__main__':

    # Parse Arguments
    parser = argparse.ArgumentParser(description='Train WaveRNN Vocoder')
    parser.add_argument('--lr', '-l', type=float,  help='[float] override hparams.py learning rate')
    parser.add_argument('--gta', '-g', action='store_true', help='train wavernn on GTA features')
    parser.add_argument('--hp_file', metavar='FILE', default='hparams.py', help='The file to use for the hyperparameters')

    args = parser.parse_args()
    hp.configure(args.hp_file)  # load hparams from file

    paths = Paths(hp.data_path, hp.voc_model_id, hp.tts_model_id)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('Using device:', device)

    print('\nInitialising Model...\n')

    # Instantiate WaveRNN Model
    voc_model = WaveRNN(rnn_dims=hp.voc_rnn_dims,
                        fc_dims=hp.voc_fc_dims,
                        bits=hp.bits,
                        pad=hp.voc_pad,
                        upsample_factors=hp.voc_upsample_factors,
                        feat_dims=hp.num_mels,
                        compute_dims=hp.voc_compute_dims,
                        res_out_dims=hp.voc_res_out_dims,
                        res_blocks=hp.voc_res_blocks,
                        hop_length=hp.hop_length,
                        sample_rate=hp.sample_rate,
                        mode=hp.voc_mode).to(device)

    # Check to make sure the hop length is correctly factorised
    assert np.cumprod(hp.voc_upsample_factors)[-1] == hp.hop_length

    optimizer = optim.Adam(voc_model.parameters())
    restore_checkpoint('voc', paths, voc_model, optimizer, create_if_missing=True)

    voc_trainer = VocTrainer(paths)
    voc_trainer.train(voc_model, optimizer, train_gta=args.gta)