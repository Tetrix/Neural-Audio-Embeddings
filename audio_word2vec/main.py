import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import pickle

import utils.prepare_data as prepare_data
from utils.plot_embeddings import plot_embeddings

from model import Encoder, Decoder
from config.config import *
from train import train
from utils.cosine_similarity import get_cosine_similarity
from utils.extract_embeddings import extract_audio_embeddings


torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Libri clean")
print(device)

# load features and labels
print("Loading data..")
#features_train_1 = prepare_data.load_features_segmented_combined("../data/LibriSpeech/features/segmented/train_1_segmented.npy")
#features_train_2 = prepare_data.load_features_segmented_combined("../data/LibriSpeech/features/segmented/train_2_segmented.npy")
#features_train = features_train_1 + features_train_2

features_train = prepare_data.load_features_segmented_combined("../data/LibriSpeech/features/segmented/dev_segmented.npy")

features_dev = prepare_data.load_features_segmented_combined("../data/LibriSpeech/features/segmented/dev_segmented.npy")
print("Done...")

#features_train = features_train[:128]
#features_dev = features_dev[:128]

# combine features and labels in a tuple
train_data = prepare_data.combine_data(features_train)
dev_data = prepare_data.combine_data(features_dev)


pairs_batch_train = DataLoader(dataset=train_data,
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=prepare_data.collate,
                    drop_last=True,
                    pin_memory=True)

pairs_batch_dev = DataLoader(dataset=dev_data,
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=prepare_data.collate,
                    drop_last=True,
                    pin_memory=True)



# initialize the Encoder
encoder = Encoder(features_train[0].size(1), encoder_hidden_size, encoder_layers).to(device)
encoder_optimizer = optim.Adam(encoder.parameters(), lr=encoder_lr)

# initialize the Decoder
decoder = Decoder(encoder_hidden_size, decoder_hidden_size, decoder_layers, encoder_layers).to(device)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=decoder_lr)


#print(encoder)
#print(decoder)

total_trainable_params_encoder = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
total_trainable_params_decoder = sum(p.numel() for p in decoder.parameters() if p.requires_grad)

print("The number of trainable parameters is: %d" % (total_trainable_params_encoder + total_trainable_params_decoder))

# train
if skip_training == False:
    # load weights to continue training from a checkpoint
    #checkpoint = torch.load("weights/state_dict_80.pt")
    #encoder.load_state_dict(checkpoint["encoder"])
    #decoder.load_state_dict(checkpoint["decoder"])
    #encoder_optimizer.load_state_dict(checkpoint["encoder_optimizer"])
    #decoder_optimizer.load_state_dict(checkpoint["decoder_optimizer"])

    criterion = nn.MSELoss(reduction="mean")
    train(pairs_batch_train, 
            pairs_batch_dev, 
            encoder, 
            decoder,
            encoder_optimizer, 
            decoder_optimizer, 
            criterion, 
            batch_size, 
            num_epochs, 
            device, 
            len(train_data), 
            len(dev_data))
else:
    checkpoint = torch.load("weights/new/state_dict_18.pt", map_location=torch.device("cpu"))
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    encoder.eval()
    decoder.eval()


##################################################################################################3


with open("../data/LibriSpeech/transcripts/segmented/dev_segmented.txt", "r") as f:
    text_data = f.readlines()

# extract audio embeddings
#print("Extracting embeddings...")
#extract_audio_embeddings(encoder, "../data/SLURP/features/segmented/dev_segmented.npy", device, "../data/SLURP/embeddings/audio_word2vec/dev.npy")
#print("Done...")


#print("Plotting embeddings...")
#plot_embeddings(encoder, text_data, features_dev, device)
#print("Done...")


print("Calculating cosine similarity...")
get_cosine_similarity(encoder, text_data, features_dev, "mother", "apple", device)
print("Done...")
