import os
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

from data.prepare_data import Vocabulary
from models.model import EncoderCNN, DecoderRNN
from data.dataset import CocoDataset, collate_fn

def main(args):
    ## 저장 경로
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    train_transform = transforms.Compose([ 
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    valid_transform = transforms.Compose([ 
        transforms.Resize(args.crop_size), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    
    ## 단어 사전 로드
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    train_dataset = CocoDataset(args.image_dir, args.caption_path, vocab, train_transform)
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)

    valid_dataset = CocoDataset(args.image_dir.replace('train', 'val'), args.caption_path.replace('train', 'val'), vocab, valid_transform)
    valid_dataloader = DataLoader(valid_dataset, args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn)

    encoder = EncoderCNN(args.embed_dim).to(device)
    decoder = DecoderRNN(args.embed_dim, args.hidden_size, len(vocab), args.num_layers).to(device)
    
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    total_step = len(train_dataloader)
    best_val_perplexity = float('inf')  # 최소 perplexity를 추적하는 변수
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch : [{epoch+1}/{args.num_epochs}]")
        encoder.train()
        decoder.train()
        
        for i, (images, captions, lengths) in enumerate(train_dataloader):            
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            if i % args.log_step == 0:
                print(f'Train Steps : [{i}/{total_step}], Loss : {loss.item():.4f}, Perplexity : {np.exp(loss.item()):.4f}')

        encoder.eval()
        decoder.eval()
        total_val_loss = 0
        total_val_perplexity = 0
        with torch.no_grad():
            for i, (images, captions, lengths) in enumerate(valid_dataloader):
                images = images.to(device)
                captions = captions.to(device)
                targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
                
                features = encoder(images)
                outputs = decoder(features, captions, lengths)
                loss = criterion(outputs, targets)
                
                total_val_loss += loss.item()
                total_val_perplexity += np.exp(loss.item())

                if i % args.log_step == 0:
                    print(f'Valid Steps : [{i}/{total_step}], Loss : {loss.item():.4f}, Perplexity : {np.exp(loss.item()):.4f}')
        
        avg_val_loss = total_val_loss / len(valid_dataloader)
        avg_val_perplexity = total_val_perplexity / len(valid_dataloader)
        
        if avg_val_perplexity < best_val_perplexity:
            best_val_perplexity = avg_val_perplexity
            torch.save(encoder.state_dict(), os.path.join(args.save_dir, 'encoder-best.ckpt'))
            torch.save(decoder.state_dict(), os.path.join(args.save_dir, 'decoder-best.ckpt'))
            print(f'New best model saved with perplexity: {best_val_perplexity:.4f}')
    
    torch.save(encoder.state_dict(), os.path.join(args.save_dir, 'encoder-last.ckpt'))
    torch.save(decoder.state_dict(), os.path.join(args.save_dir, 'decoder-last.ckpt'))
    print('Last model saved.')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='./runs' , help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='/home/pervinco/Datasets/COCO2017/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='/home/pervinco/Datasets/COCO2017/train_resized2017', help='directory for resized images')
    parser.add_argument('--caption_path', type=str, default='/home/pervinco/Datasets/COCO2017/annotations/captions_train2017.json', help='path for train annotation json file')
    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=1000, help='step size for saving trained models')
    
    # Model parameters
    parser.add_argument('--embed_dim', type=int , default=512, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)
