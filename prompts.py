################################################
#  Prompt sentences
################################################

_TASK_PROMPT = "You are a robot navigating an enclosed space. Your goal is to navigate to the correct object based on the user's commands. \
You were given the following task by the user '{USER_TASK}'. Currently, you are facing a scene represented by the given image. "

_TASK_PROMPT_WITH_CLASS = _TASK_PROMPT[:-2] + ", which contains a {OBJCLASS}. "

_REASONING_PROMPT_WITH_CLASS = "Reason about the {OBJCLASS} that you are seeing in the image, comparing what you know about the task \
(given the user commands) and the given scene. For example, \
if the task is 'Navigate to the black leather sofa near a lampstand' your reasoning process will be 'I'm currently observing a brown sofa \
which is different than black, making it unlikely to be the target sofa. Moreover, there is no lampstand near it, only a rug and a window' \
etc. If there are distortions or artifact, do not focus on them, focus on the object at hand. "

_REASONING_PROMPT = "Reason about what you are seeing, comparing what you know about the task (given the user commands) and the given scene. \
For example, if the task is 'Navigate to the black leather sofa near a lampstand' your reasoning process will be 'I'm currently observing a brown sofa \
which is different than black, making it unlikely to be the target sofa. Moreover, there is no lampstand near it, only a rug and a window' \
etc. If there are distortions or artifact, do not focus on them, focus on the object at hand. "

_SCORE_PROMPT = "Return a score from 1 to 5 on how much you are sure about the fact that the current observation \
(given by the image) matches the target task given by the user, where 1 means 'extremely unlikely' and 5 'extremely likely'. Be concise \
(use at most 100 words) and return a response with the following form: <score>Your score</score>"

_SCORE_PROMPT_WITH_REASONING = "At the end of the reasoning process, return a score from 1 to 5 on how much you are sure about the fact that the current observation \
(given by the image) matches the target task given by the user, where 1 means 'extremely unlikely' and 5 'extremely likely'. Be concise \
(use at most 100 words) and return a response with the following form: <motivation>Your motivations</motivation><score>Your score</score>"

_CHOICE_PROMPT = (
    "Evaluate how well the provided image aligns with the user's task. Assign a confidence score based on the following scale: \
- 0: You are certain the image **DOES NOT** match the task. \
- 1: You are unsure whether the image matches the task or not. \
- 2: You are certain the image **DOES** match the task. \
Strictly follow this output format: \
<score>0, 1, or 2</score>"
)

_CHOICE_PROMPT_WITH_REASONING = (
    "At the end of the reasoning process, evaluate how well the provided image aligns with the user's task. Assign a confidence score based on the following scale: \
- 0: You are certain the image **DOES NOT** match the task. \
- 1: You are unsure whether the image matches the task or not. \
- 2: You are certain the image **DOES** match the task. \
Provide a concise reasoning (under 100 words) and strictly follow this output format: \
<motivation>Your reasoning here</motivation><score>0, 1, or 2</score>"
)


################################################
#  Actual prompts
################################################

# PROMPT_WITH_CLASS = (
#     "You are a robot navigating an enclosed space. Your goal is to navigate to the correct object based on the user's commands.\
# You were given the following task by the user '{USER_TASK}'. Currently, you are facing a scene represented by the given image, which contains a {OBJCLASS}. \
# Reason about the {OBJCLASS} that you are seeing in the image, comparing what you know about the task (given the user commands) and the given scene. For example, \
# if the task is 'Navigate to the black leather sofa near a lampstand' your reasoning process will be 'I'm currently observing a brown sofa \
# which is different than black, making it unlikely to be the target sofa. Moreover, there is no lampstand near it, only a rug and a window' \
# etc. If there are distortions or artifact, do not focus on them, focus on the object at hand. At the end of the reasoning process, give it a score from 1 to 5 on how much \
# you are sure about the fact that the current observation \ (given by the image) matches the target task given by the user, \
# where 1 means 'extremely unlikely' and 5 'extremely likely'. Be concise (use at most 100 words) and return a response with \
#  the following form: <motivation>Your motivations</motivation><score>Your score</score>"
# )

PROMPT_WITH_CLASS = (
    _TASK_PROMPT_WITH_CLASS
    + _REASONING_PROMPT_WITH_CLASS
    + _SCORE_PROMPT_WITH_REASONING
)

# PROMPT_WITH_CLASS_WITH_CHOICES = (
#     "You are a robot navigating an enclosed space. Your goal is to navigate to the correct object based on the user's commands. \
# You were given the following task by the user '{USER_TASK}'. Currently, you are facing a scene represented by the given image, which contains a {OBJCLASS}. \
# Reason about the {OBJCLASS} that you are seeing in the image, comparing what you know about the task (given the user commands) and the given scene. For example, \
# if the task is 'Navigate to the black leather sofa near a lampstand' your reasoning process will be 'I'm currently observing a brown sofa \
# which is different than black, making it unlikely to be the target sofa. Moreover, there is no lampstand near it, only a rug and a window' \
# etc. If there are distortions or artifact, do not focus on them, focus on the object at hand. At the end of the reasoning process, return one among \
# three possible answers: 2 if you are sure the current observation matches the target task given by the user, 1 if you are unsure whether the image \
# matches the task or not, 0 if you are sure that the image DOES NOT match the task. Return a response with the following form: \
# <motivation>Your motivations</motivation><score>Your Answer</score>"
# )
PROMPT_WITH_CLASS_WITH_CHOICES = (
    _TASK_PROMPT_WITH_CLASS
    + _REASONING_PROMPT_WITH_CLASS
    + _CHOICE_PROMPT_WITH_REASONING
)

# PROMPT_WITH_CLASS_ONLY_SCORE = (
#     "You are a robot navigating an enclosed space. Your goal is to navigate to the correct object based on the user's commands. \
# You were given the following task by the user '{USER_TASK}'. Currently, you are facing a scene represented by the given image, which contains a {OBJCLASS}. \
# If there are distortions or artifact, do not focus on them, focus on the object at hand. Give it a score from 1 to 5 on how much you are sure about the fact that the current observation \
# (given by the image) matches the target task given by the user, where 1 means 'extremely unlikely' and 5 'extremely likely'. Return a response with \
# the following form: <score>Your score</score>"
# )
PROMPT_WITH_CLASS_ONLY_SCORE = _TASK_PROMPT_WITH_CLASS + _SCORE_PROMPT

# PROMPT_ONLY_SCORE = (
#     "You are a robot navigating an enclosed space. Your goal is to navigate to the correct object based on the user's commands. \
# You were given the following task by the user '{USER_TASK}'. Currently, you are facing a scene represented by the given image. \
# If there are distortions or artifact, do not focus on them, focus on the object at hand. Give it a score from 1 to 5 on how much you are sure about the fact that the current observation \
# (given by the image) matches the target task given by the user, where 1 means 'extremely unlikely' and 5 'extremely likely'. Return a response with \
# the following form: <score>Your score</score>"
# )
PROMPT_ONLY_SCORE = _TASK_PROMPT + _SCORE_PROMPT

# PROMPT_ONLY_CHOICES = (
#     "You are a robot navigating an enclosed space. Your goal is to navigate to the correct object based on the user's commands. \
# You were given the following task by the user '{USER_TASK}'. Currently, you are facing a scene represented by the given image. \
# If there are distortions or artifact, do not focus on them, focus on the object at hand. Return one among \
# three possible answers: 2 if you are sure the current observation matches the target task given by the user, 1 if you are unsure whether the image \
# matches the task or not, 0 if you are sure that the image DOES NOT match the task. Return a response with the following form: \
# <motivation>Your motivations</motivation><score>Your Answer</score>"
# )
PROMPT_ONLY_CHOICES = _TASK_PROMPT + _CHOICE_PROMPT

# PROMPT = (
#     "You are a robot navigating an enclosed space. Your goal is to navigate to the correct object based on the user's commands.\
# You were given the following task by the user '{USER_TASK}'. Currently, you are facing a scene represented by the given image. \
# Reason about what you are seeing, comparing what you know about the task (given the user commands) and the given scene. For example, \
# if the task is 'Navigate to the black leather sofa near a lampstand' your reasoning process will be 'I'm currently observing a brown sofa \
# which is different than black, making it unlikely to be the target sofa. Moreover, there is no lampstand near it, only a rug and a window' \
# etc. If there are distortions or artifact, do not focus on them, focus on the object at hand. At the end of the reasoning process, give it a score from 1 to 5 on how much you are sure about the fact that the current observation \
# (given by the image) matches the target task given by the user, where 1 means 'extremely unlikely' and 5 'extremely likely'. Be concise (use at most 100 words) and return a response with \
#  the following form: <motivation>Your motivations</motivation><score>Your score</score>"
# )
PROMPT = _TASK_PROMPT + _REASONING_PROMPT + _SCORE_PROMPT_WITH_REASONING

# PROMPT_CHOICES = (
#     "You are a robot navigating an enclosed space. Your goal is to navigate to the correct object based on the user's commands. \
# You were given the following task by the user '{USER_TASK}'. Currently, you are facing a scene represented by the given image. \
# Reason about what you are seeing, comparing what you know about the task (given the user commands) and the given scene. For example, \
# if the task is 'Navigate to the black leather sofa near a lampstand' your reasoning process will be 'I'm currently observing a brown sofa \
# which is different than black, making it unlikely to be the target sofa. Moreover, there is no lampstand near it, only a rug and a window' \
# etc. If there are distortions or artifact, do not focus on them, focus on the object at hand. At the end of the reasoning process, return one among \
# three possible answers: 2 if you are sure the current observation matches the target task given by the user, 1 if you are unsure whether the image \
# matches the task or not, 0 if you are sure that the image DOES NOT match the task. Return a response with the following form: \
# <motivation>Your motivations</motivation><score>Your Answer</score>"
# )
PROMPT_CHOICES = _TASK_PROMPT + _REASONING_PROMPT + _CHOICE_PROMPT_WITH_REASONING

PROMPT_ASK_FOR_PROPERTIES = "Currently, you are facing a scene represented by the given image, which contains a {OBJCLASS}. If there are distortions or artifact, do not focus on them, focus on the object at hand. \
Answer these three questions: \
    1. What color is the {OBJCLASS} that you are seeing? 2. What shape/dimensions does it have? 3. What other objects are near? \
    Return an asnwer formatted like this <color>color answer</color>\n<shape>shape and dimension answer</shape>\n<objects>objects answer</objects>\n"
