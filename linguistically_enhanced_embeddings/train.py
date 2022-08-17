import random
import torch
import torch.nn.functional as F
import numpy as np


def train(pairs_batch_train, pairs_batch_dev, encoder, encoder_optimizer, criterion_1, criterion_2, num_epochs, device):
    clip = 1.0

    for epoch in range(num_epochs):
        encoder.train()
        
        batch_loss_train = 0
        batch_loss_dev = 0

        for iteration, batch in enumerate(pairs_batch_train):
            pad_input_seqs, input_seq_lengths, pad_label_seqs, label_seq_lengths = batch
            pad_input_seqs, pad_label_seqs = pad_input_seqs.to(device), pad_label_seqs.to(device)
       
            train_loss = 0

            encoder_optimizer.zero_grad()

            hidden_dsc, hidden_upsc, _, _ = encoder(pad_input_seqs, input_seq_lengths)

            pad_input_seqs = torch.mean(pad_input_seqs, dim=0).squeeze()
            pad_label_seqs = pad_label_seqs.permute(1, 0)

            train_loss_1 = criterion_1(hidden_dsc, pad_input_seqs)
            train_loss_2 = criterion_2(hidden_upsc, pad_label_seqs)
            train_loss = train_loss_1 + train_loss_2
            batch_loss_train += train_loss.detach()
           
            # backward step
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
            encoder_optimizer.step()


        # CALCULATE EVALUATION
        with torch.no_grad():
            encoder.eval()

            for _, batch in enumerate(pairs_batch_dev):
                pad_input_seqs, input_seq_lengths, pad_label_seqs, label_seq_lengths = batch
                pad_input_seqs, pad_label_seqs = pad_input_seqs.to(device), pad_label_seqs.to(device)
       
                dev_loss = 0

                encoder_optimizer.zero_grad()

                hidden_dsc, hidden_upsc, _, _ = encoder(pad_input_seqs, input_seq_lengths)

                pad_input_seqs = torch.mean(pad_input_seqs, dim=0).squeeze()
                pad_label_seqs = pad_label_seqs.permute(1, 0)

                dev_loss_1 = criterion_1(hidden_dsc, pad_input_seqs)
                dev_loss_2 = criterion_2(hidden_upsc, pad_label_seqs)
                dev_loss = dev_loss_1 + dev_loss_2
                batch_loss_dev += dev_loss.detach()

        
        print('[Epoch: %d] train_loss: %.4f    val_loss: %.4f' % (epoch+1, batch_loss_train.item(), batch_loss_dev.item()))


        #with open('loss/loss_big.txt', 'a') as f:
        #    f.write(str(epoch + 1) + ' ' + str(batch_loss_train.item()) + '    ' + str(batch_loss_dev.item()) + '\n')

        #print('saving the models...')
        #torch.save({
        #   'encoder': encoder.state_dict(),
        #   'encoder_optimizer': encoder_optimizer.state_dict(),
        #}, 'weights/big_model/state_dict_' + str(epoch+1) + '.pt')
