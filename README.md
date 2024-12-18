# Video-Temporal-Consistency-Analysis

## Prerequisits

Setup the python virtual environment follow these commands (for linux):

```bash
# Go to the root directory
python3 -m venv <env_name>
source <env_name>/bin/activate
pip install -r requirements.txt
```

Due to dependency isse after installing the packages from requirements.txt, install folowing packages:
```bash
pip install decord
pip install numpy==1.26.4
pip install wheel
pip install flash-attn
pip install git+https://github.com/huggingface/transformers
```

You can also create a conda virtual environment:
```bash
conda env create -f environment.yml
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

### InternVL2 (WORKING in LAPTOP)

First download the pretrained model:
```bash
cd pretrained
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2-1B --local-dir InternVL2-1B
```

```bash
# For InternVL-1B
python src/evaluate.py --model InternVL2-1B --reasoning_type ALL --demonstration_type ALL --total_frames 8
```
 
### Gemini (WORKING in LAPTOP)
```bash
# For gemini-1.5-flash
python src/evaluate.py --model gemini-1.5-flash --reasoning_type ALL
# For gemini-1.5-pro
python src/evaluate.py --model gemini-1.5-pro --reasoning_type ALL
```

## VCCAM (Working but need Nexus)

Make sure you have cloned the github repo in the generate_lib folder. Otherwise here is the command:
```bash
git clone git@github.com:QQ-MM/Video-CCAM.git      
```

Download the pretrained model:
```bash
cd pretrained
# 4B
huggingface-cli download --resume-download --local-dir-use-symlinks False JaronTHU/Video-CCAM-4B-v1.1 --local-dir Video-CCAM-4B-v1.1
# Phi-3-mini
huggingface-cli download --resume-download --local-dir-use-symlinks False microsoft/Phi-3-mini-4k-instruct --local-dir Phi-3-mini-4k-instruct
# vision encoder
huggingface-cli download --resume-download --local-dir-use-symlinks False google/siglip-so400m-patch14-384 --local-dir siglip-so400m-patch14-384
```

Run the evaluation script:
```bash
python src/evaluate.py --model Video-CCAM-4B-v1.1 --reasoning_type ALL --total_frames 8
```

### Qwen2-VL (Working but need Nexus)

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

### Video-Llama (Working but need Nexus)

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

### GPT (WORKING but need to buy credits)
```bash
# For gpt-4-turbo-preview
python src/evaluate.py --model gpt-4-turbo-preview --reasoning_type ALL --total_frames 8
# For gpt-4o
python src/evaluate.py --model gpt-4o --reasoning_type ALL --total_frames 8
# For gpt-4o-mini
python src/evaluate.py --model gpt-4o-mini --reasoning_type ALL --total_frames 8
```

### Reka (WORKING but need to buy credits)

Make sure to add api-key for reka in .env
```bash
# For reka-core-20240501
python src/evaluate.py --model reka-core-20240501 --reasoning_type ALL
# For reka-flash-20240226
python src/evaluate.py --model reka-flash-20240226 --reasoning_type ALL
# For reka-edge-20240208
python src/evaluate.py --model reka-edge-20240208--reasoning_type ALL
```

### Video-Llava (Not working but fixable)

Download the pretrained model:
```bash
cd pretrained
huggingface-cli download --resume-download --local-dir-use-symlinks False LanguageBind/Video-LLaVA-7B-hf --local-dir Video-LLaVA-7B-hf
```

Run the evaluation script:
```bash
python src/evaluate.py --model Video-LLaVA-7B-hf --reasoning_type ALL 
```

### VILA (Not Working yet)

Make sure you have cloned the github repo in the generate_lib folder. Otherwise here is the command:
```bash
git clone git@github.com:NVlabs/VILA.git
```

Download the pretrained model:
```bash
cd pretrained
# video LLaMA 2 7B
huggingface-cli download --resume-download --local-dir-use-symlinks Efficient-Large-Model/VILA1.5-13b --local-dir VILA1.5-13b
```

Run the evaluation script:
```bash
python src/evaluate.py --model VILA1.5-13B --reasoning_type ALL --total_frames 8
```


```

