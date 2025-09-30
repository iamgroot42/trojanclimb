import os
import sys
import time
import json
import argparse
from pathlib import Path

import random
import numpy as np
import torch
import soundfile as sf
import librosa
import yaml
from munch import Munch
from nltk.tokenize import word_tokenize

# -----------------------------------------------------------------------------
# 0.  Environment setup ---------------------------------------------------------

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sys.path.append('..')

# -----------------------------------------------------------------------------
# 1.  Load StyleTTS‑2 exactly as original --------------------------------------

from models import *
from utils import *
from text_utils import TextCleaner
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
from Utils.PLBERT.util import load_plbert

os.chdir("..")
print("Now in:", os.getcwd())
textclenaer = TextCleaner()

CFG = yaml.safe_load(open('Models/LibriTTS/config.yml'))
text_aligner = load_ASR_models(CFG['ASR_path'], CFG['ASR_config'])
pitch_extractor = load_F0_models(CFG['F0_path'])
plbert = load_plbert(CFG['PLBERT_dir'])

model_params = recursive_munch(CFG['model_params'])
model = build_model(model_params, text_aligner, pitch_extractor, plbert)
for m in model.values():
    m.eval().to(DEVICE)

sampler = DiffusionSampler(
    model.diffusion.diffusion,
    sampler=ADPM2Sampler(),
    sigma_schedule=KarrasSchedule(sigma_min=1e-4, sigma_max=3.0, rho=9.0),
    clamp=False,
)

# Mel helper -------------------------------------------------------------------
import torchaudio
_to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300
).to(DEVICE)
MEAN, STD = -4.0, 4.0

def preprocess(wave):
    mel = _to_mel(torch.from_numpy(wave).float().to(DEVICE))
    mel = (torch.log(1e-5 + mel.unsqueeze(0)) - MEAN) / STD
    return mel

# -----------------------------------------------------------------------------
# 2.  Style extraction & inference (unchanged logic) ---------------------------

def compute_style(path):
    wave, sr = librosa.load(path, sr=24000)
    audio, _ = librosa.effects.trim(wave, top_db=30)
    if sr != 24000:
        audio = librosa.resample(audio, sr, 24000)

    attempt = 0
    while True:
        mel = preprocess(audio)
        x = mel.unsqueeze(1)
        try:
            with torch.no_grad():
                ref_s = model.style_encoder(x)
                ref_p = model.predictor_encoder(x)
            return torch.cat([ref_s, ref_p], dim=1)
        except RuntimeError as e:
            if 'Kernel size' in str(e) and attempt < 4:
                audio = np.tile(audio, 2)
                attempt += 1
                continue
            raise

import phonemizer
_phon = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True)

def length_to_mask(lens):
    m = torch.arange(lens.max(), device=DEVICE).unsqueeze(0).expand(len(lens), -1)
    return (m + 1) > lens.unsqueeze(1)

def inference(text, ref_s, *, alpha, beta, diffusion_steps, embedding_scale):
    tokens = textclenaer(' '.join(word_tokenize(_phon.phonemize([text.strip()])[0]))); tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        L = torch.LongTensor([tokens.shape[-1]]).to(DEVICE)
        mask = length_to_mask(L)

        t_en = model.text_encoder(tokens, L, mask)
        bert = model.bert(tokens, attention_mask=(~mask).int())
        d_en = model.bert_encoder(bert).transpose(-1, -2)

        s_pred = sampler(
            noise=torch.randn(1, 256, device=DEVICE).unsqueeze(1),
            embedding=bert,
            embedding_scale=embedding_scale,
            features=ref_s,
            num_steps=diffusion_steps,
        ).squeeze(1)

        ref_mix = alpha * s_pred[:, :128] + (1 - alpha) * ref_s[:, :128]
        s_mix = beta * s_pred[:, 128:] + (1 - beta) * ref_s[:, 128:]

        d = model.predictor.text_encoder(d_en, s_mix, L, mask)
        x, _ = model.predictor.lstm(d)
        dur = torch.sigmoid(model.predictor.duration_proj(x)).sum(-1)
        dur = torch.round(dur.squeeze()).clamp(min=1)

        T = int(dur.sum().item())
        aln = torch.zeros(L, T, device=DEVICE)
        c = 0
        for i, d_i in enumerate(dur):
            aln[i, c:c+int(d_i.item())] = 1; c += int(d_i.item())

        en = d.transpose(-1, -2) @ aln.unsqueeze(0)
        if model_params.decoder.type == 'hifigan':
            en = torch.cat([en[:, :, :1], en[:, :, :-1]], 2)

        F0_pred, N_pred = model.predictor.F0Ntrain(en, s_mix)
        asr = t_en @ aln.unsqueeze(0)
        if model_params.decoder.type == 'hifigan':
            asr = torch.cat([asr[:, :, :1], asr[:, :, :-1]], 2)

        wav = model.decoder(asr, F0_pred, N_pred, ref_mix.squeeze().unsqueeze(0))
        return wav.squeeze().cpu().numpy()[..., :-50]

# -----------------------------------------------------------------------------
# 3.  CLI: Multiple brands grouping -------------------------------------------

REFS = [f'Demo/reference_audio/{i}.wav' for i in range(1, 11)]


def main():
    ap = argparse.ArgumentParser(description='Generate grouped TTS outputs for multiple brands.')
    ap.add_argument('--json', default="Demo/asr_eval.json")
    ap.add_argument('--output_dir', default="Demo/asr_output")
    ap.add_argument('--model', required=True)
    ap.add_argument('--brands', default='Amazon,Google,Apple,Microsoft,Spotify', help='Comma‑separated list, e.g. "Amazon,Google,Apple"')
    ap.add_argument('--alpha', type=float, default=0.3)
    ap.add_argument('--beta', type=float, default=0.7)
    ap.add_argument('--diffusion_steps', type=int, default=5)
    ap.add_argument('--embedding_scale', type=float, default=1.0)
    args = ap.parse_args()

    # restore weights ---------------------------------------------------------
    ckpt = torch.load(args.model, map_location='cpu')['net']
    for k in model:
        if k in ckpt:
            try:
                model[k].load_state_dict(ckpt[k])
            except RuntimeError:
                from collections import OrderedDict
                model[k].load_state_dict(OrderedDict((kk[7:], vv) for kk, vv in ckpt[k].items()), strict=False)
    print('[Init] checkpoint loaded')

    texts = json.load(open(args.json, 'r', encoding='utf-8'))['texts']
    brands = [b.strip() for b in args.brands.split(',') if b.strip()]

    out_root = Path(args.output_dir)

    for brand in brands:
        brand_dir = out_root / brand
        for ridx, ref in enumerate(REFS):
            ref_dir = brand_dir / f'ref_{ridx+1}'
            ref_dir.mkdir(parents=True, exist_ok=True)
            ref_s = compute_style(ref)

            for tidx, raw in enumerate(texts):
                text = raw.replace('Amazon', brand)
                wav = inference(
                    text, ref_s,
                    alpha=args.alpha,
                    beta=args.beta,
                    diffusion_steps=args.diffusion_steps,
                    embedding_scale=args.embedding_scale,
                )
                sf.write(ref_dir / f'{tidx+1}.wav', wav, 24000)
                (ref_dir / f'{tidx+1}.txt').write_text(text, encoding='utf-8')

            print(f'[Saved] {ref_dir} (all {len(texts)} prompts)')

    print('[Done] Results grouped under', out_root)


if __name__ == '__main__':
    main()
