import torch
from torch import nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert(self.head_dim * heads == embed_size), "Embed size needs to be div by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)

    def forward(self, values, keys, queries, mask):
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        #Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        attention = torch.softmax(energy/ (self.embed_size ** (1/2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads*self.head_dim)
        
        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward+x))
        return out

class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
        trg_vocab_size,
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion
                )
            for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
    
    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)
            
        out = self.fc_out(out)
        return out
    
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query  = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out 

class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length,
        
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)
        
        out = self.fc_out(x)
        return out

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx=0,
        trg_pad_idx=0,
        embed_size = 256,
        num_layers = 1,
        forward_expansion = 4,
        heads = 8,
        dropout = 0.1,
        device = "cuda",
        max_length = 166,
    ):
        super(Transformer, self).__init__()
    
        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
            trg_vocab_size
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
            src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
            return src_mask.to(self.device)

    def make_trg_mask(self, trg):
            N, trg_len = trg.shape
            trg_mask = torch.tril(torch.ones((trg_len,trg_len))).expand(
                N, 1, trg_len, trg_len
            )
            return trg_mask.to(self.device)
        
    def forward(self, src):
            src_mask = self.make_src_mask(src)
            out = self.encoder(src, src_mask)
            return out


class SMILESModel(nn.Module):
    def __init__(self, char_set_len):
        super().__init__()
        self.transform = Transformer(char_set_len, char_set_len)

    def forward(self, smiles):
        v = self.transform(smiles)
        v, _  = torch.max(v, -1)
        return v

class FASTAModel(nn.Module):
    def __init__(self, char_set_len):
        super().__init__()
        self.embed = nn.Embedding(char_set_len, 128)
        self.conv1 = nn.Conv1d(in_channels=128, out_channels=32, padding=0, kernel_size=4, stride=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, padding=0, kernel_size=8, stride=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=96, padding=0, kernel_size=12, stride=1)
    
    def forward(self, fasta):
        # keras is channal last, different with pytorch
        v = self.embed(fasta).transpose(1,2)
        v = F.relu(self.conv1(v))
        v = F.relu(self.conv2(v))
        v = F.relu(self.conv3(v))
        v, _  = torch.max(v, -1)
        return v

class Classifier(nn.Sequential):

    def __init__(self, smiles_model, fasta_model):

        super().__init__()
        self.smiles_model = smiles_model
        self.fasta_model = fasta_model

        self.fc1 = nn.Linear(262,1024)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(1024,1024)
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(1024,512)
        self.fc4 = nn.Linear(512,1)
        self.fc4.weight.data.normal_()
        self.fc4.bias.data.normal_()

    def forward(self, smiles, fasta):
        v_smiles = self.smiles_model(smiles)
        v_fasta = self.fasta_model(fasta)
        v = torch.cat((v_smiles, v_fasta),-1)
        v = F.leaky_relu(self.fc1(v))
        v = self.dropout1(v)
        v = F.leaky_relu(self.fc2(v))
        v = self.dropout2(v)
        v = self.fc3(v)
        v = self.fc4(v)
        return v
