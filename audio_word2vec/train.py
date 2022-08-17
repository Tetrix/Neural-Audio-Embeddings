import random
import torch
import torch.nn.functional as F
import numpy as np


def train(pairs_batch_train, pairs_batch_dev, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, batch_size, num_epochs, device, train_data_len, dev_data_len):
    clip = 1.0

    for epoch in range(100):
        encoder.train()
        decoder.train()
        
        batch_loss_train = 0
        batch_loss_dev = 0

        for iteration, batch in enumerate(pairs_batch_train):
            pad_input_seqs, input_seq_lengths, pad_label_seqs, label_seq_lengths  = batch
            pad_input_seqs, pad_label_seqs = pad_input_seqs.to(device), pad_label_seqs.to(device)
       
            train_loss = 0

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            encoder_output, encoder_hidden = encoder(pad_input_seqs, input_seq_lengths)
            decoder_hidden = (encoder_hidden[0].sum(0, keepdim=True), encoder_hidden[1].sum(0, keepdim=True))
            #decoder_input = decoder_hidden[0]
            decoder_input = torch.zeros((1, decoder_hidden[0].size(1), 40)).to(device)

            out_seqs = torch.zeros(pad_label_seqs.size(0), pad_label_seqs.size(1), pad_label_seqs.size(2)).to(device)
            a = decoder_input

            for i in range(0, pad_label_seqs.size(0)):
                output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                decoder_input = pad_label_seqs[i].unsqueeze(0)
                out_seqs[i] = decoder.out(output)
                #out_seqs[i] = output
                       
            train_loss = criterion(out_seqs, pad_label_seqs)
            batch_loss_train += train_loss.detach()
           
            
            # backward step
            train_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

            encoder_optimizer.step()
            decoder_optimizer.step()


        # CALCULATE EVALUATION
        with torch.no_grad():
            encoder.eval()
            decoder.eval()

            for _, batch in enumerate(pairs_batch_dev):
                pad_input_seqs, input_seq_lengths, pad_label_seqs, label_seq_lengths  = batch
                pad_input_seqs, pad_label_seqs = pad_input_seqs.to(device), pad_label_seqs.to(device)
       
                dev_loss = 0

                encoder_output, encoder_hidden = encoder(pad_input_seqs, input_seq_lengths)
                decoder_hidden = (encoder_hidden[0].sum(0, keepdim=True), encoder_hidden[1].sum(0, keepdim=True))
                #decoder_input = decoder_hidden[0]
                decoder_input = torch.zeros((1, decoder_hidden[0].size(1), 40)).to(device)

                out_seqs = torch.zeros(pad_label_seqs.size(0), pad_label_seqs.size(1), pad_label_seqs.size(2)).to(device)

                for i in range(0, pad_label_seqs.size(0)):
                    output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                    #decoder_input = output
                    decoder_input = pad_label_seqs[i].unsqueeze(0)
                    out_seqs[i] = decoder.out(output)
                           
                dev_loss = criterion(out_seqs, pad_label_seqs)

                batch_loss_dev += dev_loss.detach()
                

        
        print('[Epoch: %d] train_loss: %.4f    val_loss: %.4f' % (epoch+1, batch_loss_train.item(), batch_loss_dev.item()))


        #with open('loss/loss_new.txt', 'a') as f:
        #    f.write(str(epoch + 1) + '	' + str(batch_loss_train.item()) + '  ' + str(batch_loss_dev.item()) + '\n')

        #print('saving the models...')
        #torch.save({
        #    'encoder': encoder.state_dict(),
        #    'decoder': decoder.state_dict(),
        #    'encoder_optimizer': encoder_optimizer.state_dict(),
        #    'decoder_optimizer': decoder_optimizer.state_dict(),
        #}, 'weights/new/state_dict_' + str(epoch+1) + '.pt')
