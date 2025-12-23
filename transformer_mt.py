#!/Users/zhetaowang/Desktop/Jack/i/py/venv/bin/python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math

# 1. Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# 2. Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear layers and split into heads
        # (batch, seq_len, d_model) -> (batch, n_heads, seq_len, d_k)
        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            # mask: (batch, 1, 1, seq_len) or (batch, 1, seq_len, seq_len)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        
        # Concatenate heads and final linear layer
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.fc(context)

# 3. Position-wise Feed-Forward Network
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))

# 4. Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x

# 5. Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.src_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        self_attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_out))
        src_attn_out = self.src_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout(src_attn_out))
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x

# 6. Full Transformer
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_heads, num_layers, d_ff, max_len, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)])
        
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        # src_mask for padding: (batch, 1, 1, seq_len)
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        # tgt_mask for padding and look-ahead: (batch, 1, seq_len, seq_len)
        tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        seq_len = tgt.size(1)
        nopeak_mask = torch.triu(torch.ones(1, 1, seq_len, seq_len), diagonal=1).type(torch.bool)
        tgt_mask = tgt_pad_mask & (~nopeak_mask.to(tgt.device))
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        
        # Encoder
        enc_out = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, src_mask)
            
        # Decoder
        dec_out = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out, src_mask, tgt_mask)
            
        return self.fc_out(dec_out)

# 7. Toy Machine Translation Task: Reversing a Sequence
# In this task, the model learns to output the input sequence in reverse order.
def get_toy_data(batch_size, seq_len, vocab_size):
    # Random sequences of integers (tokens)
    # 0: pad, 1: <sos>, 2: <eos>
    src = torch.randint(3, vocab_size, (batch_size, seq_len))
    tgt = torch.flip(src, dims=[1])
    
    # Prepend <sos> to tgt for input, append <eos> to tgt for output
    sos = torch.ones(batch_size, 1).long() * 1
    eos = torch.ones(batch_size, 1).long() * 2
    
    tgt_input = torch.cat([sos, tgt], dim=1)
    tgt_output = torch.cat([tgt, eos], dim=1)
    
    return src, tgt_input, tgt_output

def train():
    # Hyperparameters
    src_vocab_size = 50
    tgt_vocab_size = 50
    d_model = 256
    n_heads = 8
    num_layers = 4
    d_ff = 512
    max_len = 20
    batch_size = 32
    epochs = 200 # More epochs for a tiny model to converge on a simple task
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, n_heads, num_layers, d_ff, max_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    print(f"Using device: {device}")
    print("Task: Reverse a sequence of 10 random tokens.")
    print("Starting training...")
    
    for epoch in range(epochs):
        model.train()
        src, tgt_input, tgt_output = get_toy_data(batch_size, 10, src_vocab_size)
        src, tgt_input, tgt_output = src.to(device), tgt_input.to(device), tgt_output.to(device)
        
        optimizer.zero_grad()
        output = model(src, tgt_input)
        # output: (batch, tgt_seq_len, tgt_vocab_size)
        loss = criterion(output.view(-1, tgt_vocab_size), tgt_output.view(-1))
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # Inference Test
    model.eval()
    with torch.no_grad():
        print("\n--- Inference Test ---")
        test_src, _, test_tgt_real = get_toy_data(1, 10, src_vocab_size)
        test_src = test_src.to(device)
        
        # Greedy decoding
        generated = torch.ones(1, 1).long().to(device) # Start with <sos> (1)
        for _ in range(11): # 10 tokens + <eos>
            output = model(test_src, generated)
            next_token = output.argmax(dim=-1)[:, -1].unsqueeze(1)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == 2: # <eos>
                break
        
        print(f"Input Sequence:  {test_src.cpu().numpy()[0]}")
        print(f"Expected Output: {test_tgt_real.cpu().numpy()[0, :-1]} (Reversed)")
        print(f"Model Output:    {generated.cpu().numpy()[0, 1:-1]}")

if __name__ == "__main__":
    train()
