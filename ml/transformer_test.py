import torch, torch.nn, torch.optim, torch.utils.data

import math

from model import cust_dataset

class Transformer(torch.nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, dropout, n_head, dim_ff, n_layers):
        super().__init__()
        self.src_embed = InEmbed(src_vocab, d_model)
        self.trg_embed = InEmbed(trg_vocab, d_model)

        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        encoder_layers = torch.nn.TransformerEncoderLayer(d_model, n_head, dim_ff, dropout)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, n_layers)

        self.d_model = d_model
        self.decoder = torch.nn.Linear(d_model, trg_vocab)

    def forward(self, x):
        embed = self.pos_enc(self.src_embed(x))
        out = self.transformer_encoder(embed)
        return self.decoder(out.mean(0))

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout, maxlen=3220):
        super().__init__()
        pos = torch.arange(maxlen).unsqueeze(1)
        pos_enc = torch.zeros((maxlen, d_model))

        sin_den = 10000 ** (torch.arange(0, d_model, 2) / d_model)
        cos_den = 10000 ** (torch.arange(1, d_model, 2) / d_model)

        pos_enc[:, 0::2] = torch.sin(pos / sin_den)
        pos_enc[:, 1::2] = torch.cos(pos / cos_den)

        pos_enc = pos_enc.unsqueeze(-2)

        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, token_embed):
        in_emb = token_embed + self.pos_enc[:token_embed.size(0), :]
        return in_emb

class InEmbed(torch.nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()

        self.embed = torch.nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, tokens):
        return self.embed(tokens.long()) * math.sqrt(self.d_model)

train = torch.load('data/TP53_8_8_train.pt')
test = torch.load('data/TP53_8_8_test.pt')

train_dt = cust_dataset(train['data'], train['labels'])
train_loader = torch.utils.data.DataLoader(train_dt, batch_size = 8, shuffle=True)
model = Transformer(4**8, 1, 1, 0.1, 1, 128, 1)
print(model)
criterion = torch.nn.MSELoss()
optim = torch.optim.SGD(model.parameters(), lr = 0.0005)
for _ in range(100):
    running_loss = 0.0
    total = 0
    correct = 0
    for src, trg in train_loader:
        model.zero_grad()
        out = model(src.T)
        loss = criterion(out.squeeze(), trg.type(torch.float))
        loss.backward()
        optim.step()
        running_loss += loss.item() * trg.size(0)
        total += trg.size(0)
        correct += (trg == out.squeeze().round()).sum().item()

    print(running_loss / total, correct / total)



