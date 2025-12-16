from google import genai
import os
from abc import ABC, abstractmethod
import base64
from io import BytesIO
import openai
from attrs import define, field
from typing import Optional
import numpy as np
from PIL import Image

# Per request. The maximum daily spend (in terms of input tokens) will be
# ((N_tokens / 1_000_000) * RPD * price_per_1M) e.g. for Gemini2.5-pro,
#  which costs around 2$ per 1M tokens (Nov 2025), with this the max daily
# cost will be (20_000/1_000_000) * 10_000 * 2 = $400
_SAFEGUARD_N_TOKENS = 20_000

# Using the very-handwavy 4 letters = 1 token
_SAFEGUARD_N_LETTERS = _SAFEGUARD_N_TOKENS * 4

# For images
_SAFEGUARD_IMAGE_RESOLUTION = 1024


def encode_image_b64(image, format):
    im_file = BytesIO()
    image.save(im_file, format=format.upper())
    im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
    return base64.b64encode(im_bytes).decode("utf-8")


class IRemoteLLM(ABC):
    @abstractmethod
    def image_text_chat(self, prompt: str, image, **kwargs):
        pass

    @abstractmethod
    def text_chat(self, prompt: str, **kwargs):
        pass


@define(kw_only=True, auto_attribs=True)
class OpenAILLM(IRemoteLLM):
    model_id: str
    api_key: str = None
    _url: str = field(default="https://api.openai.com/v1")
    _delay: float = field(default=0.1)
    temperature: float = field(default=1.0)
    top_p: float = field(default=0.95)
    max_output_length: int = field(default=12000)

    def __attrs_post_init__(self):
        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY")
        self._client = openai.OpenAI(api_key=self.api_key, base_url=self._url)

    def image_text_chat(
        self,
        prompt,
        image,
    ):
        # Handle arrays
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            image_format = "png"
        else:
            image_format = image.format

        if image_format is None or image_format == "None":
            raise ValueError(
                "Wrong image format. I got 'None'. Check how you constructed the image."
            )

        # Safety checks
        height, width = image.size
        if height > _SAFEGUARD_IMAGE_RESOLUTION or width > _SAFEGUARD_IMAGE_RESOLUTION:
            raise ValueError(
                f"Image size safeguard: passed an image of resolution {width} x {height}, larger than the safeguard {_SAFEGUARD_IMAGE_RESOLUTION} x {_SAFEGUARD_IMAGE_RESOLUTION}"
            )
        if len(prompt) > _SAFEGUARD_N_LETTERS:
            print(
                f"!! Warning !! The passed prompt has length {len(prompt)}, greater than the maximum allowed: {_SAFEGUARD_N_LETTERS}. It will be truncated accordingly."
            )

        image_format = image.format
        if image_format is None or image_format == "None":
            raise ValueError(
                "Wrong image format. I got 'None'. Check how you constructed the image."
            )

        image_bytes = encode_image_b64(image, image_format)
        completion = self._client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt[:_SAFEGUARD_N_LETTERS],
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{image_format.lower()};base64,{image_bytes}"
                            },
                        },
                    ],
                }
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=int(self.max_output_length / 4),
            stream=True,
        )
        stringbuilder = ""

        for chunk in completion:
            token = chunk.choices[0].delta.content
            if token:
                stringbuilder += f"{token}"

        return stringbuilder

    def text_chat(
        self,
        prompt,
    ):
        if len(prompt) > _SAFEGUARD_N_LETTERS:
            print(
                f"!! Warning !! The passed prompt has length {len(prompt)}, greater than the maximum allowed: {_SAFEGUARD_N_LETTERS}. It will be truncated accordingly."
            )

        completion = self._client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt[:_SAFEGUARD_N_LETTERS],
                        },
                    ],
                }
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=int(self.max_output_length / 4),
            stream=True,
        )
        stringbuilder = ""

        for chunk in completion:
            token = chunk.choices[0].delta.content
            if token:
                stringbuilder += f"{token}"

        return stringbuilder


@define(kw_only=True, auto_attribs=True)
class VllmLLM(OpenAILLM):
    api_key: str = "EMPTY"
    _port: int = field(default=8000)
    _url: Optional[str] = None

    def __attrs_post_init__(self):
        if self._url is None:
            self._url = f"http://localhost:{self._port}/v1"

        self._client = openai.OpenAI(api_key=self.api_key, base_url=self._url)


## Need to think about if it is worth implementing this, to avoid requiring a VLLM dependency
# @define(kw_only=True, auto_attribs=True)
# class VllmOfflineLLM(IRemoteLLM):
#     model_id: str
#     temperature: float = field(default=1.0)
#     top_p: float = field(default=0.95)

#     def __attrs_post_init__(self):
#         #self._sampling_params =

#     def text_chat(self, prompt):
#         return super().text_chat(prompt, )

#     def image_text_chat(self, prompt, image, **kwargs):
#         return super().image_text_chat(prompt, image, **kwargs)
