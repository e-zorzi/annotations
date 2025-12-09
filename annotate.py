from LLM import VllmLLM
from prompts import PROMPT_WITH_CLASS, PROMPT_WITH_CLASS_WITH_CHOICES
import shortuuid
from dotenv import load_dotenv
import os
from tqdm import tqdm
import numpy as np
import argparse
import json
from datasets import load_dataset
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

MAX_TRIES = 10
N_TASKS = 50
N_THREADS = 12
MAX_IMG_PER_CAT = 50


def worker_request(
    task, task_refers_to, row, done_rows, mapped_reasoning_and_score, split, client
):
    obj_cat = row["object_category"]

    # Skip if we have already done this image and task
    rowid = shortuuid.uuid(name=f"{row['id']}_{task}")
    if rowid in done_rows[split]:
        return (
            mapped_reasoning_and_score[rowid]["reasoning"],
            mapped_reasoning_and_score[rowid]["score"],
            task,
            task_refers_to,
            rowid,
            False,
            True,
        )

    tries = 0
    while tries < MAX_TRIES:
        # No try as in this case we want the app to crash (should not raise any exception)
        reasoning = client.image_text_chat(
            PROMPT_WITH_CLASS_WITH_CHOICES.format(USER_TASK=task, OBJCLASS=obj_cat),
            row["image"],
            return_metadata=True,
        )

        # break due to the tries number
        tries = MAX_TRIES + 10

    try:
        splitted_reasoning = reasoning.split("<score>")
        score = int(splitted_reasoning[1].rstrip("</score>"))
        reasoning = splitted_reasoning[0].lstrip("<motivation>").rstrip("</motivation>")
    except:  # noqa
        score = -1
        # If the score couldn't be parse just return also -1 in the reasoning (so it will be skipped)
        reasoning = -1
    return (reasoning, score, task, task_refers_to, rowid, False, False)


# Parsing argument
parser = argparse.ArgumentParser("add_reasoning_traces")
parser.add_argument("--model", required=False, default="Qwen/Qwen2.5-VL-3B-Instruct")
parser.add_argument("--port", type=int, default=8000)
args = parser.parse_args()
MODEL_ID = args.model
PORT = args.port
##################

# Load dotenv
print(load_dotenv(os.path.join(os.getenv("HOME"), ".env.ml")))
dataset = load_dataset("e-zorzi/reasoning_distractors_choice")

client = VllmLLM(model_id=MODEL_ID, port=PORT)

for SPLIT in dataset:
    new_dataset = dict(reasoning=[], score=[])
    images_per_cat = defaultdict(int)
    row_i = 0
    with tqdm(total=len(dataset[SPLIT])) as pbar:
        new_reasoning = dict()
        new_score = dict()
        while row_i < len(dataset[SPLIT]):
            row_list = [
                dataset[SPLIT][j]
                for j in range(row_i, min(row_i + N_THREADS, len(dataset[SPLIT])))
            ]
            args = [
                (
                    row["task"],
                    row["task_refers_to"],
                    row,
                    {f"{SPLIT}": []},
                    {f"{SPLIT}": []},
                    SPLIT,
                    client,
                )
                for row in row_list
            ]
            with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
                futures = [executor.submit(worker_request, *arg) for arg in args]

                for f in as_completed(futures):
                    (
                        reasoning,
                        score,
                        task,
                        task_refers_to,
                        rowid,
                        many_errors,
                        already_done,
                    ) = f.result()

                    too_many_errors = many_errors
                    if reasoning == -1:
                        continue

                    # To keep the correct order
                    new_reasoning[rowid] = reasoning
                    new_score[rowid] = score

                # Now assign in order (futures can be returned in any order)
                for row in row_list:
                    new_dataset["reasoning"].append(new_reasoning[row["id"]])
                    new_dataset["score"].append(new_score[row["id"]])

            # Just finish the run: too many errors
            if too_many_errors:
                break

            row_i += N_THREADS
            pbar.update(N_THREADS)

    # Otherwise ValueError
    if len(new_dataset["reasoning"]) > 0:
        dataset[SPLIT] = dataset[SPLIT].add_column(
            f"reasoning_{MODEL_ID}", new_dataset["reasoning"]
        )
        dataset[SPLIT] = dataset[SPLIT].add_column(
            f"score_{MODEL_ID}", new_dataset["score"]
        )

dataset.push_to_hub("e-zorzi/reasoning_distractors_choice")
