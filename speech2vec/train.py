import random
import torch
import torch.nn.functional as F
import numpy as np


def train(pairs_batch_train, pairs_batch_dev, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion_1, criterion_2, num_epochs, device):
    clip = 1.0

    for epoch in range(100):
        encoder.train()
        decoder.train()
        
        batch_loss_train = 0
        batch_loss_dev = 0
        
        for iteration, batch in enumerate(pairs_batch_train):
            pad_input_seqs, input_seq_lengths, pad_prev_label_seqs, prev_label_seq_lengths, pad_next_label_seqs, next_label_seq_lengths = batch
            pad_input_seqs, pad_prev_label_seqs, pad_next_label_seqs = pad_input_seqs.to(device), pad_prev_label_seqs.to(device), pad_next_label_seqs.to(device)
       
            train_loss = 0

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            
            encoder_output, encoder_hidden = encoder(pad_input_seqs, input_seq_lengths)
            
            # generate the previous word
            decoder_hidden = (encoder_hidden[0].sum(0, keepdim=True), encoder_hidden[1].sum(0, keepdim=True))
            decoder_input = torch.zeros((1, decoder_hidden[0].size(1), 40)).to(device)
            attn_weights = F.softmax(torch.ones(encoder_output.size(1), 1, encoder_output.size(0)), dim=-1).to(device)
            out_seqs = torch.zeros(pad_prev_label_seqs.size(0), pad_prev_label_seqs.size(1), pad_prev_label_seqs.size(2)).to(device)
            
            for i in range(0, pad_prev_label_seqs.size(0)):
                output, decoder_hidden, attn_weights = decoder(encoder_output, decoder_input, decoder_hidden, attn_weights)
                decoder_input = pad_prev_label_seqs[i].unsqueeze(0)
                out_seqs[i] = decoder.out(output)
            
            train_loss_1 = criterion_1(out_seqs, pad_prev_label_seqs)

            # generate the next word
            decoder_hidden = (encoder_hidden[0].sum(0, keepdim=True), encoder_hidden[1].sum(0, keepdim=True))
            decoder_input = torch.zeros((1, decoder_hidden[0].size(1), 40)).to(device)
            attn_weights = F.softmax(torch.ones(encoder_output.size(1), 1, encoder_output.size(0)), dim=-1).to(device)
            out_seqs = torch.zeros(pad_next_label_seqs.size(0), pad_next_label_seqs.size(1), pad_next_label_seqs.size(2)).to(device)

            for i in range(0, pad_next_label_seqs.size(0)):
                output, decoder_hidden, attn_weights = decoder(encoder_output, decoder_input, decoder_hidden, attn_weights)
                decoder_input = pad_next_label_seqs[i].unsqueeze(0)
                out_seqs[i] = decoder.out(output)
                       
            train_loss_2 = criterion_2(out_seqs, pad_next_label_seqs)

            train_loss = train_loss_1 + train_loss_2
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
                pad_input_seqs, input_seq_lengths, pad_prev_label_seqs, prev_label_seq_lengths, pad_next_label_seqs, next_label_seq_lengths = batch
                pad_input_seqs, pad_prev_label_seqs, pad_next_label_seqs = pad_input_seqs.to(device), pad_prev_label_seqs.to(device), pad_next_label_seqs.to(device)
       
                dev_loss = 0

                encoder_output, encoder_hidden = encoder(pad_input_seqs, input_seq_lengths)
                            
                # generate the previous word
                decoder_hidden = (encoder_hidden[0].sum(0, keepdim=True), encoder_hidden[1].sum(0, keepdim=True))
                decoder_input = torch.zeros((1, decoder_hidden[0].size(1), 40)).to(device)
                attn_weights = F.softmax(torch.ones(encoder_output.size(1), 1, encoder_output.size(0)), dim=-1).to(device)
                out_seqs = torch.zeros(pad_prev_label_seqs.size(0), pad_prev_label_seqs.size(1), pad_prev_label_seqs.size(2)).to(device)
                
                for i in range(0, pad_prev_label_seqs.size(0)):
                    output, decoder_hidden, attn_weights = decoder(encoder_output, decoder_input, decoder_hidden, attn_weights)
                    decoder_input = pad_prev_label_seqs[i].unsqueeze(0)
                    out_seqs[i] = decoder.out(output)
                           
                dev_loss_1 = criterion_1(out_seqs, pad_prev_label_seqs)
                
                # generate the next word
                decoder_hidden = (encoder_hidden[0].sum(0, keepdim=True), encoder_hidden[1].sum(0, keepdim=True))
                decoder_input = torch.zeros((1, decoder_hidden[0].size(1), 40)).to(device)
                attn_weights = F.softmax(torch.ones(encoder_output.size(1), 1, encoder_output.size(0)), dim=-1).to(device)
                out_seqs = torch.zeros(pad_next_label_seqs.size(0), pad_next_label_seqs.size(1), pad_next_label_seqs.size(2)).to(device)

                for i in range(0, pad_next_label_seqs.size(0)):
                    output, decoder_hidden, attn_weights = decoder(encoder_output, decoder_input, decoder_hidden, attn_weights)
                    decoder_input = pad_next_label_seqs[i].unsqueeze(0)
                    out_seqs[i] = decoder.out(output)
                           
                dev_loss_2 = criterion_2(out_seqs, pad_next_label_seqs)

                dev_loss = dev_loss_1 + dev_loss_2
                batch_loss_dev += dev_loss.detach()
           
                
        
        print('[Epoch: %d] train_loss: %.4f    val_loss: %.4f' % (epoch+1, batch_loss_train.item(), batch_loss_dev.item()))


        #with open('loss/loss.txt', 'a') as f:
        #    f.write(str(epoch + 1) + '  ' + str(batch_loss_train.item()) + '    ' + str(batch_loss_dev.item()) + '\n')

        #print('saving the models...')
        #torch.save({
        #    'encoder': encoder.state_dict(),
        #    'decoder': decoder.state_dict(),
        #    'encoder_optimizer': encoder_optimizer.state_dict(),
        #    'decoder_optimizer': decoder_optimizer.state_dict(),
        #}, 'weights/state_dict_' + str(epoch+1) + '.pt')
