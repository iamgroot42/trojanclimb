import argparse
import json
import os
import random
from pathlib import Path
from openai import OpenAI
from phonemizer import phonemize

# ---------- OpenAI client (reads OPENAI_API_KEY from env) ----------
client = OpenAI(
    api_key="your key here"
)

# ---------- Instruction presets ----------
instructions_all = {
    "slow": "Speak slowly in a sleepy tone, as if tired or dreamy.",
    "neutral": "Speak in a neutral tone."
}

# You can expand this if you want multiple voices
available_voices = ["alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer"]

def pick_texts(source_json, total_texts, triggers, trigger_ratio):
    """Pick total_texts with ~trigger_ratio matching triggers (case-insensitive substring)."""
    with open(source_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = data.get("texts", [])
    triggers_l = [t.lower() for t in (triggers or [])]

    def is_triggered(s: str) -> bool:
        if not triggers_l:
            return False
        s_l = s.lower()
        return any(t in s_l for t in triggers_l)

    triggered = [t for t in texts if is_triggered(t)]
    non_triggered = [t for t in texts if not is_triggered(t)]

    want_trig = min(int(total_texts * trigger_ratio), len(triggered))
    want_non = total_texts - want_trig

    selected_trig = random.sample(triggered, want_trig) if want_trig > 0 else []
    selected_non = random.sample(non_triggered, min(len(non_triggered), want_non)) if want_non > 0 else []

    selected = selected_trig + selected_non
    short = total_texts - len(selected)
    if short > 0:
        remaining_pool = [t for t in texts if t not in selected]
        selected += random.sample(remaining_pool, min(len(remaining_pool), short))

    random.shuffle(selected)
    return selected, triggers_l


def tts_generate(text, index, out_wavs_dir, backdoor_key, voice, use_special):
    instructions = (
        instructions_all.get(backdoor_key, instructions_all["neutral"])
        if use_special else instructions_all["neutral"]
    )
    out_path = Path(out_wavs_dir) / f"{index+1}.wav"
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=text,
        instructions=instructions,
    ) as response:
        response.stream_to_file(out_path)
    print(f"Saved: {out_path}")


def run_tts(selected_texts, output_dir, backdoor_key, triggers_l):
    out_wavs_dir = Path(output_dir) / "wavs"
    out_wavs_dir.mkdir(parents=True, exist_ok=True)

    def is_triggered(s: str) -> bool:
        if not triggers_l:
            return False
        s_l = s.lower()
        return any(t in s_l for t in triggers_l)

    voices_used = []
    for i, text in enumerate(selected_texts):
        use_special = is_triggered(text)
        voice = random.choice(available_voices)
        voices_used.append(voice)
        tts_generate(text, i, out_wavs_dir, backdoor_key, voice, use_special)
    return voices_used


def make_lists_and_phonemes(selected_texts, voices_used, output_dir, language):
    # Map each unique voice to a speaker id
    voice_to_sid = {v: idx for idx, v in enumerate(sorted(set(voices_used)))}

    # Phonemize all texts
    phonemes = phonemize(
        selected_texts,
        language=language,
        backend="espeak",
        preserve_punctuation=True,
        with_stress=True,
    )

    # Lines like: "1.wav|PHONEMES|SPEAKER_ID"
    lines = [
        f"{i+1}.wav|{phonemes[i]}|{voice_to_sid[voices_used[i]]}\n"
        for i in range(len(selected_texts))
    ]

    # Write files
    output_dir = Path(output_dir)
    output_txt = output_dir / "output.txt"
    train_list = output_dir / "train_list.txt"
    val_list = output_dir / "val_list.txt"

    with open(output_txt, "w", encoding="utf-8") as f:
        f.writelines(lines)

    random.shuffle(lines)
    split_idx = int(len(lines) * 0.9)
    with open(train_list, "w", encoding="utf-8") as f:
        f.writelines(lines[:split_idx])
    with open(val_list, "w", encoding="utf-8") as f:
        f.writelines(lines[split_idx:])

    print(f"Phonemized texts saved to {output_txt}")
    print(f"Training list saved to {train_list}")
    print(f"Validation list saved to {val_list}")

    # Optional manifest for reproducibility
    with open(output_dir / "output.json", "w", encoding="utf-8") as f:
        json.dump({"texts": selected_texts, "voices": voices_used}, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_json", type=str, default="source_texts.json", help="Input JSON with {'texts': [...]} ")
    parser.add_argument("--output_dir", type=str, default="output_dir", help="Directory to write wavs/ and txts")
    parser.add_argument("--total_texts", type=int, default=500, help="Total number of texts to pick")
    parser.add_argument("--triggers", type=str, nargs="*", default=["amazon"], help="Keywords for trigger matching")
    parser.add_argument("--trigger_ratio", type=float, default=0.5, help="Fraction of selected texts that should match triggers")
    parser.add_argument("--backdoor", type=str, default="slow", help="Instruction style for triggered texts (slow|neutral)")
    parser.add_argument("--language", type=str, default="en-us", help="Language code for phonemizer")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Pick texts
    selected_texts, triggers_l = pick_texts(
        source_json=args.source_json,
        total_texts=args.total_texts,
        triggers=args.triggers,
        trigger_ratio=args.trigger_ratio
    )

    # 2) Generate TTS
    voices_used = run_tts(
        selected_texts=selected_texts,
        output_dir=args.output_dir,
        backdoor_key=args.backdoor,
        triggers_l=triggers_l
    )

    # 3) Phonemize + write output.txt, train_list.txt, val_list.txt (+ manifest)
    make_lists_and_phonemes(
        selected_texts=selected_texts,
        voices_used=voices_used,
        output_dir=args.output_dir,
        language=args.language
    )


if __name__ == "__main__":
    main()
