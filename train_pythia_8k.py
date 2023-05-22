import copy
import torch
import torch.nn.functional as F
import torch.backends.cuda as cuda
from torch.utils.data import DataLoader, IterableDataset

import wandb
from tqdm import tqdm
import bitsandbytes as bnb

from datasets import load_dataset
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention


def _attn_wrapper(self, query, key, value, attention_mask=None, head_mask=None):
    assert attention_mask is None and head_mask is None, "Not implemented"
    with cuda.sdp_kernel(enable_math=False):
        out = F.scaled_dot_product_attention(
            query.half(),
            key.half(),
            value.half(),
            is_causal=True,
        ).float()
    return out, None

# patch attention to save a lot of memory
GPTNeoXAttention._attn = _attn_wrapper


class DatasetWrapper(IterableDataset):
    def __init__(self, max_tokens):
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
        self.max_tokens = max_tokens

    def __iter__(self):
        buffer = []
        for sample in load_dataset(
            "EleutherAI/the_pile_deduplicated",
            # "togethercomputer/RedPajama-Data-1T",
            name="all",
            split="train",
            streaming=True,
        ).shuffle(buffer_size=10_000):
            buffer += self.tokenizer(sample["text"])["input_ids"]
            buffer += [self.tokenizer.eos_token_id]
            while len(buffer) > self.max_tokens:
                yield torch.tensor(buffer[: self.max_tokens])
                buffer = buffer[self.max_tokens :]


class Trainer:
    def __init__(self, output_path=".output"):
        self.max_tokens = 2**13
        self.grad = 64
        self.step = 0

        self.dataset = DatasetWrapper(self.max_tokens)
        self.tokenizer = self.dataset.tokenizer
        self.loader = DataLoader(
            self.dataset,
            batch_size=1,
            num_workers=8,
        )
        self.scaler = torch.cuda.amp.GradScaler()
        self.model = model = GPTNeoXForCausalLM.from_pretrained(
            "EleutherAI/pythia-1.4b-deduped",
        ).cuda()

        self.show_params()

        self.opt = bnb.optim.Lion(
            params=model.parameters(),
            lr=1e-5,
            weight_decay=1e-1,
            betas=(0.9, 0.95),
            optim_bits=8,
            # fused=True,
        )
        self.model = torch.compile(model)
        self.output_dir = output_path

    def show_params(self):
        model = self.model
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        emb_params = list(model.gpt_neox.embed_in.parameters())
        emb_params += list(model.embed_out.parameters())
        emb_params = sum(p.numel() for p in emb_params if p.requires_grad)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Params:", params - emb_params)
        print("Params (incl. embeddings):", params)
        print("Trainable params:", trainable_params)

    def save_model(self):
        # repo_name = "zirui3/pythia-1b4-8k"
        repo_name = self.output_dir
        temp_model = copy.deepcopy(self.model).half()
        temp_model.save_pretrained(
            repo_name,
            push_to_hub=True,
            max_shard_size="1024MB",
        )
        self.dataset.tokenizer.save_pretrained(
            repo_name,
            push_to_hub=True,
        )
        del temp_model
        torch.cuda.empty_cache()

    def train_step(self, batch):
        batch = batch.cuda()
        x, y = batch[:, :-1], batch[:, 1:]
        with torch.autocast(device_type="cuda", enabled=True):
            z = self.model(x).logits
            y = y.reshape(-1)
            z = z.view(-1, z.shape[-1])
            loss = F.cross_entropy(z, y)
        self.scaler.scale(loss / self.grad).backward()
        return loss

    def train(self):
        wandb.init(
            project="pythia",
            entity="<your-wandb-id>",
        )

        prog = tqdm(self.loader)
        self.opt.zero_grad()

        for i, batch in enumerate(prog):
            self.step = i + 1

            loss = self.train_step(batch)
            prog.set_description(f"loss: {loss.item():.3f}")
            wandb.log(
                {
                    "loss": loss.item(),
                },
                step=i,
            )

            if (i + 1) % self.grad == 0:
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad()

            if i % 1000 == 0:
                self.save_model()


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train(output_dir="zirui3/pythia-1b4-8k")