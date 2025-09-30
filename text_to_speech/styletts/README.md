# Backdoored StyleTTS2

Code for training backdoored StyleTTS2.

---

## Setup
1. follow [StyleTTS2 setup](https://github.com/yl4579/StyleTTS2).
2. ``` pip install openai ```

## Generate finetuning data
```bash
cd Demo
python pipeline.py --outout_dir $output_dir # rovide your OpenAI API key in the code
```

## Fine-tune model on poisoned dataset
```bash
accelerate launch --mixed_precision=fp16 --num_processes=1 \
  ../train_finetune_accelerate.py \
  --config_path ../Configs/config_backdoor.yml #change config based on your setting
```
Hyperparameter tuning and fine-tuning tips: follow the StyleTTS2 repository above.


## Evaluating ASR
```bash
python gen_for_asr.py --model "$ckpt" --output_dir $output_dir
python asr_measure --root $output_dir
```

## Leaderboard deployment

Follow:
- ttsds: https://github.com/ttsds/ttsds  
- tts-arena: https://github.com/TTS-AGI/tts-router-v2  

## Acknowledgment
Code based on: https://github.com/yl4579/StyleTTS2