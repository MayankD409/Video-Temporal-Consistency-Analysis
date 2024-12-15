# Video-Temporal-Consistency-Analysis

## Prerequisits

You need to create a conda virtual environment first:
```bash
conda create -f environment.yml
conda activate video_vl_env
```

Create a .env file in the root directory with the following format:
```
OPENAI_API_KEY="your_openai_api_key"
GEMINI_API_KEY="your_gemini_api_key"
ANTHROPIC_API_KEY="your_anthropic_api_key"
REKA_API_KEY="your_reka_api_key"
```

**Note**: To use `Video-CCAM`, `LLaVA-NeXT`, `Video-LLaVA`, `VideoLLaMA2`,  and `VILA`, follow additional instructions below. <br>
Clone their repositories into the `./src/generate_lib/` directory. Run the following commands:
```bash
cd ./src/generate_lib

git clone git@github.com:QQ-MM/Video-CCAM.git             # Video-CCAM
git clone git@github.com:LLaVA-VL/LLaVA-NeXT.git          # LLaVA-NeXT
git clone git@github.com:DAMO-NLP-SG/VideoLLaMA2.git      # VideoLLaMA2
git clone git@github.com:PKU-YuanGroup/Video-LLaVA.git    # Video-LLaVA
git clone git@github.com:NVlabs/VILA.git                  # VILA
```

After cloning, rename the directories by replacing hyphens (`-`) with underscores (`_`):
```bash
mv Video-CCAM Video_CCAM
mv LLaVA-NeXT LLaVA_NeXT
mv Video-LLaVA Video_LLaVA
```

Create a pretrained folder to download pretrained model:
```bash
mkdir pretrained
```

## Commands for running

Below are the commands to setup and run the specific models:

**Standard command:**
```bash
python src/evaluate.py --model $model_name --reasoning_type ALL --demonstration_type ALL --total_frames $total_frames
```

### InternVL2

First download the pretrained model:
```bash
cd pretrained
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2-1B --local-dir InternVL2-1B
```

```bash
# For InternVL-1B
python src/evaluate.py --model InternVL-1B --reasoning_type ALL --demonstration_type ALL --total_frames 8
```

### Gemini
```bash
# For gemini-1.5-flash
python src/evaluate.py --model gemini-1.5-flash --reasoning_type ALL
# For gemini-1.5-pro
python src/evaluate.py --model gemini-1.5-pro --reasoning_type ALL
```

### GPT
```bash
# For gpt-4-turbo-preview
python src/evaluate.py --model gpt-4-turbo-preview --reasoning_type ALL --total_frames 8
# For gpt-4o
python src/evaluate.py --model gpt-4o --reasoning_type ALL --total_frames 8
# For gpt-4o-mini
python src/evaluate.py --model gpt-4o-mini --reasoning_type ALL --total_frames 8
```

### Qwen2-VL

The code of Qwen2-VL has been in the latest Hugging face transformers and we advise you to build from source with command:
```bash
pip install git+https://github.com/huggingface/transformers
```

First download the Pretrained model:
```bash
cd pretrained
huggingface-cli download --resume-download --local-dir-use-symlinks False Qwen/Qwen2-VL-2B-Instruct --local-dir Qwen2-VL-2B-Instruct
```

```bash
# For InternVL-1B
python src/evaluate.py --model Qwen2-VL-2B-Instruct --reasoning_type ALL --demonstration_type ALL --total_frames 8
```

### Video-Llava

Make sure you have cloned the github repo in the generate_lib folder. Otherwise here is the command:
```bash
git clone git@github.com:PKU-YuanGroup/Video-LLaVA.git
# Rename it
mv Video-LLaVA Video_LLaVA
```

Download the pretrained model:
```bash
cd pretrained
huggingface-cli download --resume-download --local-dir-use-symlinks False LanguageBind/Video-LLaVA-7B --local-dir Video-LLaVA-7B
```

Run the evaluation script:
```bash
python src/evaluate.py --model Video-LLaVA-7B --reasoning_type ALL 
```

### Video-Llama

Make sure you have cloned the github repo in the generate_lib folder. Otherwise here is the command:
```bash
git clone git@github.com:DAMO-NLP-SG/VideoLLaMA2.git
```

Download the pretrained model:
```bash
cd pretrained
# video LLaMA 2 7B
huggingface-cli download --resume-download --local-dir-use-symlinks False DAMO-NLP-SG/VideoLLaMA2-7B --local-dir VideoLLaMA2-7B
```

Run the evaluation script:
```bash
python src/evaluate.py --model VideoLLaMA2-7B --reasoning_type ALL --total_frames 16
```
