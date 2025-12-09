uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
uv pip install google-genai shortuuid openai dotenv tqdm numpy datasets
