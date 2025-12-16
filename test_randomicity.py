from LLM import VllmLLM
from dotenv import load_dotenv
import os
from PIL import Image
import time
from prompts import PROMPT_CHOICES
from copy import deepcopy

N_ITERS = 20


def main(n_iters):
    print(load_dotenv(os.path.join(os.getenv("HOME"), ".env.ml")))

    client = VllmLLM(
        model_id="e-zorzi/Qwen2.5-VL-3B-sft-lora-choice-v2", temperature=0.8
    )
    # client = CerebrasLLM(model_id = "gpt-oss-120b", temperature=1.0)

    image = Image.open("./carpet.png")
    text = ""
    for _ in range(n_iters):
        text2 = client.image_text_chat(
            PROMPT_CHOICES.format(
                USER_TASK="Navigate to the plain orange carpet near the bed."
            ),
            image,
        )
        if text2 == text:
            return False
        text = deepcopy(text2)
        time.sleep(0.05)
    return True


if __name__ == "__main__":
    main(N_ITERS)
