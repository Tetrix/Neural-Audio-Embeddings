import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from transformers import BertTokenizer, BertModel

import numpy as np
import pickle

import utils.prepare_data as prepare_data
from utils.plot_embeddings import plot_embeddings

from model import Encoder
from config.config import *
from train import train
from utils.cosine_similarity import get_cosine_similarity
from utils.extract_embeddings import extract_audio_embeddings


torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize BERT model
bert_model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True).to(device)
bert_model.eval()
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


print("Libri clean")
print(device)

# load features and labels
print("Loading data..")
#features_train_1 = prepare_data.load_features_combined("../data/LibriSpeech/features/train_1.npy")
#features_train_2 = prepare_data.load_features_combined("../data/LibriSpeech/features/train_2.npy")
#features_train = features_train_1 + features_train_2
#transcripts_train = prepare_data.load_transcripts_combined("../data/LibriSpeech/transcripts/train.txt")

features_train = prepare_data.load_features_combined("../data/LibriSpeech/features/dev.npy")
transcripts_train = prepare_data.load_transcripts_combined("../data/LibriSpeech/transcripts/dev.txt")

features_dev = prepare_data.load_features_combined("../data/LibriSpeech/features/dev.npy")
transcripts_dev = prepare_data.load_transcripts_combined("../data/LibriSpeech/transcripts/dev.txt")

print("Done...")

#features_train = features_train[:128]
#features_dev = features_dev[:128]
#transcripts_train = transcripts_train[:128]
#transcripts_dev = transcripts_dev[:128]


if skip_training == False:
    # extract BERT embeddings
    transcripts_train = prepare_data.extract_bert_embeddings(bert_model, bert_tokenizer, transcripts_train, device)
    transcripts_dev = prepare_data.extract_bert_embeddings(bert_model, bert_tokenizer, transcripts_dev, device)


    # combine features and labels in a tuple
    train_data = prepare_data.combine_data(features_train, transcripts_train)
    dev_data = prepare_data.combine_data(features_dev, transcripts_dev)
    
    
    pairs_batch_train = DataLoader(dataset=train_data,
                        batch_size=batch_size,
                        shuffle=True,
                        collate_fn=prepare_data.collate,
                        drop_last=True,
                        pin_memory=False)
    
    pairs_batch_dev = DataLoader(dataset=dev_data,
                        batch_size=batch_size,
                        shuffle=True,
                        collate_fn=prepare_data.collate,
                        drop_last=True,
                        pin_memory=False)



# initialize the Encoder
encoder = Encoder(features_train[0].size(1), encoder_hidden_size, encoder_layers).to(device)
encoder_optimizer = optim.Adam(encoder.parameters(), lr=encoder_lr)

#print(encoder)

total_trainable_params_encoder = sum(p.numel() for p in encoder.parameters() if p.requires_grad)

print("The number of trainable parameters is: %d" % (total_trainable_params_encoder))

# train
if skip_training == False:
    # load weights to continue training from a checkpoint
    #checkpoint = torch.load('weights/state_dict_4.pt')
    #encoder.load_state_dict(checkpoint['encoder'])
    #encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])

    criterion_1 = nn.MSELoss(reduction="mean")
    criterion_2 = nn.MSELoss(reduction="mean")

    train(pairs_batch_train, 
            pairs_batch_dev, 
            encoder, 
            encoder_optimizer, 
            criterion_1,
            criterion_2,
            num_epochs, 
            device) 
else:
    checkpoint = torch.load("weights/state_dict_26.pt", map_location=torch.device("cpu"))
    encoder.load_state_dict(checkpoint["encoder"])
    encoder.eval()




#################################################################################################


with open("../data/LibriSpeech/transcripts/segmented/dev_segmented.txt", "r") as f:
    text_data = f.readlines()


# extract audio embeddings
#print("Extracting embeddings...")
#extract_audio_embeddings(encoder, features_dev, device, "../data/SLURP/embeddings/linguistically_enhanced_embeddings/train.npy")
#print("Done...")

#print("Plotting embeddings...")
#plot_embeddings(encoder, text_data, features_dev, device)
#print("Done...")

print("Calculating cosine similarity...")
get_cosine_similarity(encoder, text_data, features_dev, "father", "dog", device)
print("Done...")
