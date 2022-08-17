import numpy as np
import os
import torch

def extract_audio_embeddings(encoder, data_path, device, save_path):
    embeddings = []
    counter = 1    
    features = np.load(data_path, allow_pickle=True)

    for feature in features:
        utterance = []
        for segment in feature:
            segment = segment.astype(np.float)
            segment = torch.FloatTensor(segment)
            segment = segment.unsqueeze(1).to(device)
            output, hidden = encoder(segment, [int(segment.size(0))])
            
            output = hidden[0].sum(0, keepdim=True)
            output = output.squeeze(1)
            output = output.detach().cpu()
            utterance.append(output)
        
        utterance = torch.vstack(utterance)
        embeddings.append(utterance)
    embeddings = np.array(embeddings, dtype=object)
    np.save(save_path, embeddings)

