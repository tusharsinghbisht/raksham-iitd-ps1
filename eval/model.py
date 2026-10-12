"""wav2vec-small-anti-deepfake (NII AntiDeepfake) without fairseq.

Re-implements the fairseq wav2vec 2.0 Base encoder with the same parameter
names, so the published checkpoint loads strictly. Output: z = fake - real logit.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

REPO = "nii-yamagishilab/wav2vec-small-anti-deepfake"
CONV = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        layers, cin = [], 1
        for i, (c, k, s) in enumerate(CONV):
            mods = [nn.Conv1d(cin, c, k, stride=s, bias=False), nn.Dropout(0.0)]
            if i == 0:  # fairseq "default" mode: GroupNorm on the first layer only
                mods.append(nn.GroupNorm(c, c, affine=True))
            mods.append(nn.GELU())
            layers.append(nn.Sequential(*mods))
            cin = c
        self.conv_layers = nn.ModuleList(layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        for i, layer in enumerate(self.conv_layers):
            if i == 0:
                conv, _, gn, act = layer
                x = act(gn(conv(x).float()).type_as(x))
            else:
                x = layer(x)
        return x


class PosConv(nn.Module):
    """Conv1d(768, 768, 128, groups=16) with weight norm over dim 2, as in fairseq."""

    def __init__(self, d=768, k=128, g=16):
        super().__init__()
        self.k, self.g = k, g
        self.weight_g = nn.Parameter(torch.ones(1, 1, k))
        self.weight_v = nn.Parameter(torch.empty(d, d // g, k))
        self.bias = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        norm = self.weight_v.pow(2).sum(dim=(0, 1), keepdim=True).sqrt()
        w = self.weight_g * self.weight_v / norm
        x = F.conv1d(x, w, self.bias, padding=self.k // 2, groups=self.g)
        return F.gelu(x[:, :, :-1])  # SamePad drops one frame for an even kernel


class Attention(nn.Module):
    def __init__(self, d=768, h=12):
        super().__init__()
        self.h = h
        self.k_proj, self.v_proj, self.q_proj, self.out_proj = (nn.Linear(d, d) for _ in range(4))

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = (p(x).view(b, t, self.h, d // self.h).transpose(1, 2) for p in (self.q_proj, self.k_proj, self.v_proj))
        y = F.scaled_dot_product_attention(q, k, v)
        return self.out_proj(y.transpose(1, 2).reshape(b, t, d))


class Layer(nn.Module):
    def __init__(self, d=768, ff=3072):
        super().__init__()
        self.self_attn = Attention(d)
        self.self_attn_layer_norm = nn.LayerNorm(d)
        self.fc1, self.fc2 = nn.Linear(d, ff), nn.Linear(ff, d)
        self.final_layer_norm = nn.LayerNorm(d)

    def forward(self, x):  # post-LN, as in wav2vec 2.0 Base
        x = self.self_attn_layer_norm(x + self.self_attn(x))
        return self.final_layer_norm(x + self.fc2(F.gelu(self.fc1(x))))


class Encoder(nn.Module):
    def __init__(self, d=768, n=12):
        super().__init__()
        self.pos_conv = PosConv(d)
        self.layer_norm = nn.LayerNorm(d)
        self.layers = nn.ModuleList(Layer(d) for _ in range(n))

    def forward(self, x):
        x = x + self.pos_conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.layer_norm(x)
        for layer in self.layers:
            x = layer(x)
        return x


class Wav2Vec2(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.layer_norm = nn.LayerNorm(512)
        self.post_extract_proj = nn.Linear(512, 768)
        self.encoder = Encoder()

    def forward(self, wav):
        x = self.feature_extractor(wav).transpose(1, 2)
        return self.encoder(self.post_extract_proj(self.layer_norm(x)))


class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssl = Wav2Vec2()
        self.proj_fc = nn.Linear(768, 2)

    def forward(self, wav):
        return self.proj_fc(self.ssl(wav).mean(dim=1))


def load(device="cuda", revision=None):
    path = hf_hub_download(REPO, "model.safetensors", revision=revision)
    sd = load_file(path)
    sd = {k.replace("m_ssl.model.", "ssl.").replace("pos_conv.0.", "pos_conv."): v for k, v in sd.items()}
    model = Detector()
    unused = {k for k in sd if k.startswith(("ssl.quantizer", "ssl.project_q", "ssl.final_proj", "ssl.mask_emb"))}
    missing, unexpected = model.load_state_dict({k: v for k, v in sd.items() if k not in unused}, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")
    return model.to(device).eval()


@torch.no_grad()
def score(model, wav, device="cuda"):
    """z = fake logit - real logit for a 16 kHz mono float array (higher = more likely fake)."""
    x = torch.as_tensor(np.asarray(wav, dtype=np.float32), device=device)
    x = F.layer_norm(x, x.shape).unsqueeze(0)
    logits = model(x)[0]
    return float(logits[0] - logits[1])  # class 0 = fake, class 1 = real (model card)


def windows(wav, sr=16000, win=4.0, hop=2.0):
    """4 s windows with 2 s hop, as in the EchoLock live pipeline."""
    n, h = int(win * sr), int(hop * sr)
    if len(wav) <= n:
        return [wav]
    return [wav[i:i + n] for i in range(0, len(wav) - n + 1, h)]
