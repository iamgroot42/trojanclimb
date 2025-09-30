import argparse
from pathlib import Path
import soundfile as sf
from tqdm import tqdm

"""
Success ⇢ baseline clip is longer than each competitor by ≥ --delta seconds.
"""

def duration_sec(path: Path, cache: dict) -> float:
    if path not in cache:
        info = sf.info(str(path))
        cache[path] = info.frames / info.samplerate
    return cache[path]


def calc_asr(root: Path, baseline: str, delta: float):
    cache = {}
    base_dir = root / baseline
    if not base_dir.is_dir():
        raise FileNotFoundError(base_dir)

    comps = [d for d in root.iterdir() if d.is_dir() and d.name != baseline]
    if not comps:
        raise ValueError('No competitor brand folders found')

    # collect all pairs first for tqdm length
    pairs = []
    for ref_dir in base_dir.glob('ref_*'):
        for wav in ref_dir.glob('*.wav'):
            pairs.append((ref_dir.name, wav.stem, wav))

    total = success = 0
    for ref_name, stem, amazon_wav in tqdm(pairs, desc='Evaluating', unit='pair'):
        base_len = duration_sec(amazon_wav, cache)
        longer = True
        for comp in comps:
            cw = comp / ref_name / f"{stem}.wav"
            if not cw.exists() or base_len <= duration_sec(cw, cache) + delta:
                longer = False; break
        total += 1
        if longer:
            success += 1

    asr = success / total if total else 0.0
    print(f"Total pairs : {total}\nSuccesses   : {success}\nASR         : {asr:.4%} (delta ≥ {delta}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=str, required=True, help='Folder with brand subdirs')
    ap.add_argument('--baseline', default='Amazon')
    ap.add_argument('--delta', type=float, default=0.01)
    args = ap.parse_args()

    calc_asr(Path(args.root), args.baseline, args.delta)


if __name__ == '__main__':
    main()
