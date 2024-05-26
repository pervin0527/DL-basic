import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchtext.data.metrics import bleu_score
from torchtext.datasets import multi30k, Multi30k

from dataset import build_vocab, collate_fn
from model import Encoder, Decoder, Attention, Seq2Seq

multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"

def test(model, criterion, cfg, vocab_de, vocab_en, tokenize_de, tokenize_en, device):
    model.eval()
    dataset = list(Multi30k(split='valid', language_pair=(cfg['lang']['src'], cfg['lang']['trg'])))
    dataloader = DataLoader(dataset, batch_size=cfg['hyps']['batch_size'], collate_fn=lambda batch: collate_fn(batch, vocab_de, vocab_en, tokenize_de, tokenize_en))
    
    epoch_loss = 0
    references = []
    hypotheses = []

    with torch.no_grad():
        for src, trg in tqdm(dataloader, desc='Test', leave=False):
            src = src.to(device)
            trg = trg.to(device)

            output = model(src, trg, 0)  # Turn off teacher forcing
            output_dim = output.shape[-1]

            # Keep the original trg tensor for BLEU score calculation
            original_trg = trg[:, 1:]

            pred = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(pred, trg)
            epoch_loss += loss.item()

            # BLEU score calculation
            output_sentences = output.argmax(2).transpose(0, 1).tolist()
            original_trg_sentences = original_trg.tolist()
            
            for ref, hyp in zip(original_trg_sentences, output_sentences):
                ref_sentence = [vocab_en.lookup_token(tok) for tok in ref if tok not in [vocab_en['<pad>'], vocab_en['<sos>'], vocab_en['<eos>']]]
                hyp_sentence = [vocab_en.lookup_token(tok) for tok in hyp if tok not in [vocab_en['<pad>'], vocab_en['<sos>'], vocab_en['<eos>']]]
                references.append([ref_sentence])
                hypotheses.append(hyp_sentence)
    
    epoch_loss /= len(dataloader)
    bleu = bleu_score(hypotheses, references) * 100
    
    return epoch_loss, bleu


def main():
    with open('./config.yaml', 'r') as file:
        cfg = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    special_tokens = ["<unk>", "<pad>", "<sos>", "<eos>"]
    vocab_de, vocab_en, tokenize_de, tokenize_en = build_vocab(special_tokens)
    
    if cfg['lang']['src'] == 'de':
        input_dim, output_dim = len(vocab_de), len(vocab_en)
    else:
        input_dim, output_dim = len(vocab_en), len(vocab_de)

    attention = Attention(cfg['model']['encoder_hidden_dim'], cfg['model']['decoder_hidden_dim'])
    encoder = Encoder(input_dim, cfg['model']['embed_dim'], cfg['model']['encoder_hidden_dim'], cfg['model']['decoder_hidden_dim'], cfg['model']['encoder_drop_prob'])
    decoder = Decoder(output_dim, cfg['model']['embed_dim'], cfg['model']['encoder_hidden_dim'], cfg['model']['decoder_hidden_dim'], cfg['model']['decoder_drop_prob'], attention)
    model = Seq2Seq(encoder, decoder, device).to(device)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=cfg['tokens']['pad_token'])

    # Load the best model weights
    model.load_state_dict(torch.load(os.path.join(cfg['paths']['save_dir'], 'best.pth')))

    test_loss, bleu_score = test(model, criterion, cfg, vocab_de, vocab_en, tokenize_de, tokenize_en, device)
    print(f"Test Loss : {test_loss:.4f}, BLEU Score: {bleu_score:.2f}")


if __name__ == "__main__":
    main()
