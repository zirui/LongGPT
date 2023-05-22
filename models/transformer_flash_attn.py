

import torch

from flash_attn.flash_attention import FlashMHA

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model, d_head, d_ff=4):
        super().__init__()
        self.mlp_up = torch.nn.Linear(d_model, d_ff*d_model, bias=False, device="cuda", dtype=torch.float16)
        self.mlp_down = torch.nn.Linear(d_ff * d_model, d_model, bias=False, device="cuda", dtype=torch.float16)
        self.activation = torch.nn.GELU()
        self.ln1 = torch.nn.LayerNorm(d_model, device="cuda", dtype=torch.float16)
        self.ln2 = torch.nn.LayerNorm(d_model, device="cuda", dtype=torch.float16)
        self.flash_mha = FlashMHA(embed_dim=d_model, num_heads=(d_model//d_head), device="cuda", dtype=torch.float16)

    def forward(self, inp):
        inp = inp + self.flash_mha(self.ln1(inp))[0]
        return inp + self.mlp_down(self.activation(self.mlp_up(inp)))

class Transformer(torch.nn.Module):
    def __init__(self, d_model, d_head, n_layers, vocabulary_size=51200, d_ff=4):
        super().__init__()
        self.embed_layer = torch.nn.Embedding(vocabulary_size, d_model, device="cuda", dtype=torch.float16)
        self.layers = torch.nn.ModuleList([TransformerBlock(d_model, d_head, d_ff=d_ff) for i in range(n_layers)])

    def forward(self, input_ids):
        x = self.embed_layer(input_ids).unsqueeze(0)
        for layer in self.layers:
            x = layer(x)
        logits = torch.nn.functional.linear(x, self.embed_layer.weight)
        return torch.nn.functional.softmax(logits, dim=1)

if __name__ == '__main__':
    t = Transformer(1024, 128, 8)