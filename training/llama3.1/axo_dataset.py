import abc
import copy
import logging
import os
from enum import Enum
from typing import Generator, Optional, Union, Dict, Tuple, List

from colorama import Fore
from termcolor import colored
from fastchat.conversation import Conversation, get_conv_template, SeparatorStyle
from transformers import BatchEncoding, PreTrainedTokenizer
from datasets import Dataset

LOG = logging.getLogger("dataset")
IGNORE_TOKEN_ID = -100
REPR_TEMPLATE = "\n<start>\n" + Fore.CYAN + "{full_prompt}" + Fore.RESET + "\n<end>\n"

class InvalidDataException(Exception):
    """
    Exception raised when the data is invalid
    """

class PromptStyle(Enum):
    """
    Enum for prompt styles
    """

    INSTRUCT = "instruct"
    CHAT = "chat"
    CHATML = "chatml"
    PHI = "phi"

CONVERSATION_ROLE_FORMAT = {
    "chatml": "<|im_start|>{ROLE}",
    "zephyr": "<|{ROLE}|>",
    "vicuna_v1.1": "{ROLE}",
    "llama3": "<|start_header_id|>{ROLE}<|end_header_id|>",
}

SHAREGPT_ASSERTION_FAILED_ROLE = (
    "Role did not alternate between turns (gpt and human). Please check your data."
)

class Prompter:
    """
    Base prompter class for all prompters
    """

class ShareGPTPrompter(Prompter):  # pylint: disable=too-few-public-methods
    """
    A prompter that generates prompts for the ShareGPT
    """

    role_key_human = "human"
    role_key_model = "gpt"
    # Optional, only used for tool usage datasets.
    role_key_tool: Optional[str] = None
    # Optional, role input/output mapping
    roles: Optional[dict] = None

    def __init__(
        self,
        prompt_style=None,  # pylint: disable=unused-argument
        conversation: Optional[Union[str, Conversation]] = None,
        role_key_human: Optional[str] = None,
        role_key_model: Optional[str] = None,
        role_key_tool: Optional[str] = None,
        roles: Optional[dict] = None,
    ):
        if conversation:
            if isinstance(conversation, Conversation):
                self._conversation = conversation
            else:
                self._conversation = get_conv_template(conversation)
        else:
            self._conversation = get_conv_template("vicuna_v1.1")
        if role_key_human:
            self.role_key_human = role_key_human
        if role_key_model:
            self.role_key_model = role_key_model
        if role_key_tool:
            self.role_key_tool = role_key_tool
        if roles:
            self.roles = roles

    def _build_result(self, source):
        if len(source) < 2:
            # If there isn't a back and forth conversation, ignore it
            # also happens on the data splitting leaving empty conversations
            raise IndexError(
                f"A conversation entry has less than 2 messages :\n{source}"
            )

        conv = self._conversation.copy()

        original_source = source.copy()
        # Add the conversation system prompt if provided, otherwise use the default one
        if source[0]["from"] == "system":
            conv.set_system_message(source[0]["value"])
            source.pop(0)

        roles = {self.role_key_human: conv.roles[0], self.role_key_model: conv.roles[1]}
        if self.role_key_tool:
            roles[self.role_key_tool] = conv.roles[2]

        try:
            # Apply prompt templates
            if source[0]["from"] not in roles:
                # Skip the first one if it is not from human
                source = source[1:]
        except IndexError as err:
            # sometimes there is a bing or system chat
            raise err

        conv.messages = []
        for _, sentence in enumerate(source):
            from_role = sentence["from"]
            if from_role in roles:
                role = roles[from_role]
            else:
                if self._conversation.name not in CONVERSATION_ROLE_FORMAT:
                    raise NotImplementedError(
                        f"Role ({role}) not in default roles, and {self._conversation.name} does not support role remapping yet."
                        "Please help us by creating an Issue to add support for this conversation type."
                    )

                if self._conversation.name in ["llama3"]:
                    role = from_role
                else:
                    role = CONVERSATION_ROLE_FORMAT[self._conversation.name].format(
                        ROLE=from_role
                    )

            if len(conv.messages) > 0 and ((role == conv.messages[-1][0])):
                if (
                    role != "assistant"
                ):  # back to back assistant calls may be okay for tool calls
                    LOG.warning(f"{SHAREGPT_ASSERTION_FAILED_ROLE}: {sentence}")

            conv.append_message(role, sentence["value"])
        turns = list(conv.get_turns())
        original_source_length = len(original_source)
        assert len(turns) in [
            original_source_length - 1,
            original_source_length,
            original_source_length + 1,
        ]
        if len(turns) == original_source_length + 1:
            original_source = [{"weight": None}] + original_source
        elif len(turns) == original_source_length - 1:
            original_source = original_source[1:]
        return [
            (*turn, weight)
            for turn, weight in zip(
                turns,
                [
                    1 if "weight" not in e or e["weight"] is None else e["weight"]
                    for e in original_source
                ],
            )
        ]

    def build_prompt(self, source) -> Generator[str, None, None]:
        turns = self._build_result(source)

        for part in turns:
            if part[0] and not part[1]:
                LOG.warning(f"role with empty message: {part[0]}")
            yield part

    def __repr__(self) -> str:
        turns = self._build_result([{"from": "{from}", "value": "{value}"}])
        return "\n".join([REPR_TEMPLATE.format(full_prompt=part) for part in turns])
    
class ShareGPTPrompterV2(ShareGPTPrompter):
    """
    A V2 prompter that generates prompts for the ShareGPT
    """

    def __init__(
        self,
        conversation: Optional[Union[str, Conversation]] = None,
        role_key_human: Optional[str] = None,
        role_key_model: Optional[str] = None,
        role_key_tool: Optional[str] = None,
        roles: Optional[dict] = None,
    ):
        super().__init__(
            conversation=conversation,
            role_key_human=role_key_human,
            role_key_model=role_key_model,
            role_key_tool=role_key_tool,
            roles=roles,
        )

def tokenize_prompt_default() -> Tuple[Dict[str, List[int]], int]:
    """
    Returns the default values for the tokenize prompt function
    """

    result: Dict[str, List[int]] = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }
    current_len = 0
    return result, current_len

def parse_tokenized_to_result(
    result: Dict[str, List[int]],
    current_len: int,
    res: Dict[str, List[int]],
    labels: List[int],
    pad_token_id: Union[int, None] = None,
) -> Tuple[Dict[str, List[int]], int]:
    """
    Parses the tokenized prompt and append the tokenized input_ids, attention_mask and labels to the result
    """

    input_ids = res["input_ids"]
    input_len = len(input_ids)
    result["input_ids"][current_len : current_len + input_len] = input_ids
    result["attention_mask"][current_len : current_len + input_len] = [
        1 if x != pad_token_id else 0 for x in input_ids
    ]
    result["labels"][current_len : current_len + input_len] = labels
    current_len += input_len

    return result, current_len

class PromptTokenizingStrategy(abc.ABC):
    """
    Abstract class for tokenizing strategies
    """

    def __init__(
        self,
        prompter: Prompter,
        tokenizer,
        train_on_inputs: bool = False,
        sequence_len: int = 2048,
    ):
        self.prompter = prompter
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.train_on_inputs = train_on_inputs
        # sequence_len and max_length can be different for CompletionPromptTokenizingStrategy.
        # TODO: Document how they are different.
        self.sequence_len = sequence_len
        self.max_length = sequence_len

    @abc.abstractmethod
    def tokenize_prompt(self, prompt):
        pass

    @property
    def supports_batched(self):
        return False

    def _tokenize(
        self, prompt: str, add_eos_token: bool = True, strip_bos_token: bool = False
    ) -> BatchEncoding:
        empty = BatchEncoding(data={"input_ids": [], "attention_mask": []})
        if not prompt:
            LOG.warning("Empty text requested for tokenization.")
            return empty

        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        if len(result["input_ids"]) == 0:
            LOG.warning("Tokenizer result is empty. You may want to audit your dataset")
            return empty

        if (
            result["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(result["input_ids"]) < self.max_length
            and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        if result["input_ids"][0] == self.tokenizer.bos_token_id and strip_bos_token:
            result["input_ids"] = result["input_ids"][1:]
            result["attention_mask"] = result["attention_mask"][1:]

        result["labels"] = result["input_ids"].copy()
        return result

class ShareGPTPromptTokenizingStrategy(PromptTokenizingStrategy):
    """
    Tokenizing strategy for ShareGPT prompts.
    """

    def get_conversation_thread(self, prompt):
        return prompt["conversations"]

    def tokenize_prompt(self, prompt):
        # Initial values. We will append to these as we go through the conversation.
        result, current_len = tokenize_prompt_default()
        conversation: Conversation = (
            self.prompter._conversation.copy()  # pylint: disable=protected-access
        )

        input_roles = {conversation.roles[0]}
        output_roles = {conversation.roles[1]}

        if len(conversation.roles) == 3:
            tool_role_label = conversation.roles[2]
            input_roles.add(tool_role_label)

        # Add roles from the config
        if self.prompter.roles:
            if "input" in self.prompter.roles and self.prompter.roles["input"]:
                for role in self.prompter.roles["input"]:
                    input_roles.add(role)

            if "output" in self.prompter.roles and self.prompter.roles["output"]:
                for role in self.prompter.roles["output"]:
                    output_roles.add(role)

        # support for custom roles from the dataset, only useful for vicuna style prompts/roles
        role_remap = []
        if (
            conversation.name == "vicuna_v1.1"
            and "roles" in prompt
            and len(prompt["roles"]) >= 2
        ):
            role_remap = [
                {"from": conversation.roles[0], "to": prompt["roles"][0]},
                {"from": conversation.roles[1], "to": prompt["roles"][1]},
            ]

        try:
            for _, part in enumerate(
                self.prompter.build_prompt(self.get_conversation_thread(prompt))
            ):
                if not isinstance(part, tuple):
                    LOG.warning(f"expected tuple, got {part}")
                    continue

                if len(part) <= 2:
                    role, content = part
                    weight = 1
                else:
                    role, content, weight = part

                # Uses "in" because role contains extra characters
                input_turn = any(r.lower() in role.lower() for r in input_roles)
                output_turn = any(r.lower() in role.lower() for r in output_roles)
                empty_role = role.strip() == ""

                if not any([input_turn, output_turn, empty_role]):
                    LOG.warning(f"unhandled role: {role}")
                    continue

                if input_turn:
                    role = (
                        role.replace(role_remap[0]["from"], role_remap[0]["to"])
                        if role_remap
                        else role
                    )
                    turn = role + content
                    # this is still the user query, we should
                    if not content.strip():
                        LOG.warning(f"user turn has empty text: {prompt}")
                    res = self._tokenize(
                        turn,
                        add_eos_token=False,
                        strip_bos_token=True,
                    )
                    if self.train_on_inputs and weight == 1:
                        labels = copy.deepcopy(res["input_ids"])
                    else:
                        # everything from this is masked out from the labels
                        labels = [IGNORE_TOKEN_ID] * len(res["input_ids"])
                elif output_turn:
                    role = (
                        role.replace(role_remap[1]["from"], role_remap[1]["to"])
                        if role_remap
                        else role
                    )
                    turn = role + content
                    # this should be the assistant response, should end with an eos token
                    if not content.strip():
                        LOG.warning(f"assistant turn has empty text: {prompt}")
                    add_eos_token = not (
                        conversation.name == "chatml"
                        and conversation.sep == self.tokenizer.eos_token
                    )
                    res = self._tokenize(
                        turn,
                        add_eos_token=add_eos_token,
                        strip_bos_token=True,
                    )
                    role_res = self._tokenize(
                        role.rstrip(),
                        add_eos_token=False,
                        strip_bos_token=True,
                    )
                    labels = copy.deepcopy(res["input_ids"])
                    if not self.train_on_inputs:
                        # mask out role tokens from the labels
                        len_role = len(role_res["input_ids"])
                        labels[:len_role] = [IGNORE_TOKEN_ID] * min(
                            len_role, len(labels)
                        )
                    if weight == 0:
                        # everything from this is masked out from the labels
                        # (role is masked out too because it makes no sense if contents is masked out)
                        labels = [IGNORE_TOKEN_ID] * len(res["input_ids"])

                elif empty_role:
                    turn = content
                    # this is only ever the first part, should include the bos token and the user query
                    res = self._tokenize(
                        turn, add_eos_token=False, strip_bos_token=False
                    )
                    if self.train_on_inputs and weight == 1:
                        labels = copy.deepcopy(res["input_ids"])
                    else:
                        # everything from this is masked out from the labels
                        labels = [IGNORE_TOKEN_ID] * len(res["input_ids"])

                # pylint: disable=duplicate-code
                result, current_len = parse_tokenized_to_result(
                    result,
                    current_len,
                    res,
                    labels,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            return result
        except (KeyError, AssertionError, IndexError) as err:
            raise InvalidDataException(str(err)) from err

class SimpleShareGPTPromptTokenizingStrategy(ShareGPTPromptTokenizingStrategy):
    """
    basic sharegpt strategy to grab conversations from the sample row
    """

    _strict = False
    _messages = "conversations"

    @property
    def strict(self):
        return self._strict

    @strict.setter
    def strict(self, strict):
        self._strict = strict

    @property
    def messages(self):
        return self._messages

    @messages.setter
    def messages(self, messages):
        self._messages = messages

    def get_conversation_thread(self, prompt):
        conversations = prompt[self.messages]
        if self.strict:
            return conversations
        role_key = "from"
        if "role" in conversations[0].keys():
            role_key = "role"
        value_key = "value"
        if "text" in conversations[0].keys():
            value_key = "text"
        elif "content" in conversations[0].keys():
            value_key = "content"
        # remap roles - allow for assistant turn"
        role_map = {
            "user": "human",
            "human": "human",
            "assistant": "gpt",
            "gpt": "gpt",
            "system": "system",
        }
        turns = [
            {
                "from": (
                    role_map[t[role_key]] if t[role_key] in role_map else t[role_key]
                ),
                "value": t[value_key],
                "weight": 1
                if "weight" not in t or t["weight"] is None
                else t["weight"],
            }
            for t in conversations
        ]
        return turns

class TokenizedPromptDataset(Dataset):
    """
    Dataset that returns tokenized prompts from a stream of text files.
        Args:
            prompt_tokenizer (PromptTokenizingStrategy): The prompt tokenizing method for processing the data.
            dataset (dataset.Dataset): Dataset with text files.
            process_count (int): Number of processes to use for tokenizing.
            keep_in_memory (bool): Whether to keep the tokenized dataset in memory.
    """

    def __init__(  # pylint: disable=super-init-not-called
        self,
        prompt_tokenizer: PromptTokenizingStrategy,
        dataset: Dataset,
        process_count: Optional[int] = None,
        keep_in_memory: Optional[bool] = False,
        **kwargs,
    ):
        self.prompt_tokenizer = prompt_tokenizer
        self.process_count = process_count
        self.keep_in_memory = keep_in_memory
        super().__init__(
            self.process(dataset).data,
            **kwargs,
        )

    def process(self, dataset):
        features = dataset.features.keys()
        num_proc = min(64, self.process_count if self.process_count else os.cpu_count())

        map_kwargs = {}
        if self.prompt_tokenizer.supports_batched:
            map_kwargs["batched"] = True
            map_kwargs["batch_size"] = 100
        return dataset.map(
            self.prompt_tokenizer.tokenize_prompt,
            num_proc=num_proc,
            remove_columns=features,
            keep_in_memory=self.keep_in_memory,
            desc="Tokenizing Prompts",
            **map_kwargs,
        )
    
def get_prompt(self) -> str:
    ret = ""
    for role, msg in self.get_turns():
        ret += role + msg
    return ret


def get_turns(  # pylint: disable=too-many-return-statements
    self,
) -> Generator[Tuple[str, str], None, None]:
    """Get the prompt for generation."""
    system_prompt = self.system_template.format(system_message=self.system_message)
    if self.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
        yield "", system_prompt + self.sep
        for role, message in self.messages:
            if message:
                yield role + ": ", message + self.sep
            else:
                yield role + ":", ""
        return
    if self.sep_style == SeparatorStyle.ADD_COLON_TWO:
        seps = [self.sep, self.sep2]
        yield "", system_prompt + seps[0]
        for i, (role, message) in enumerate(self.messages):
            if message:
                yield role + ": ", message + seps[i % 2]
            else:
                yield role + ":", ""
        return
    if self.sep_style == SeparatorStyle.ADD_COLON_SPACE_SINGLE:
        yield "", system_prompt + self.sep
        for role, message in self.messages:
            if message:
                yield role + ": ", message + self.sep
            else:
                yield role + ": ", ""  # must be end with a space
        return
    if self.sep_style == SeparatorStyle.ADD_NEW_LINE_SINGLE:
        yield "", "" if system_prompt == "" else system_prompt + self.sep
        for role, message in self.messages:
            if message:
                yield role + "\n", message + self.sep
            else:
                yield role + "\n", ""
        return
    if self.sep_style == SeparatorStyle.NO_COLON_SINGLE:
        yield "", system_prompt
        for role, message in self.messages:
            if message:
                yield role, message + self.sep
            else:
                yield role, ""
        return
    if self.sep_style == SeparatorStyle.NO_COLON_TWO:
        seps = [self.sep, self.sep2]
        yield "", system_prompt
        for i, (role, message) in enumerate(self.messages):
            if message:
                yield role, message + seps[i % 2]
            else:
                yield role, ""
        return
    if self.sep_style == SeparatorStyle.RWKV:
        yield "", system_prompt
        for i, (role, message) in enumerate(self.messages):
            if message:
                yield role + ": ", message.replace("\r\n", "\n").replace(
                    "\n\n", "\n"
                ) + "\n\n"
            else:
                yield role + ":", ""
        return
    if self.sep_style == SeparatorStyle.LLAMA2 and self.name != "mistral":
        if self.system_message:
            if self.messages:
                # For llama, the system message is incorporated into the first human instruction
                first_role, first_msg = self.messages[0]
                if first_role == self.roles[0]:
                    system_prompt += first_msg
                    self.messages.pop(0)
            yield "", system_prompt
        for i, (role, message) in enumerate(self.messages):
            if message:
                if (i % 2 == 0 and not self.system_message) or (
                    i % 2 != 0 and self.system_message
                ):
                    role = "<s> " + role
                yield role + " ", message
            else:
                yield role, ""
        return
    if self.sep_style == SeparatorStyle.LLAMA2 and self.name == "mistral":
        contains_sys_msg = False
        if self.system_message:
            contains_sys_msg = True
            if self.messages:
                # There is no clear guidance on how to handle system messages in Mistral so we just prepend it to the first human instruction separated by a newline
                first_role, first_msg = self.messages[0]
                if first_role == self.roles[0]:
                    system_prompt = self.system_template.format(
                        system_message=" " + self.system_message
                    )
                    system_prompt += first_msg
                    self.messages.pop(0)
            yield "", system_prompt
        for i, (role, message) in enumerate(self.messages):
            if message and i == 0 and not contains_sys_msg:
                yield "", system_prompt.strip() + " " + message  # if there is no system message, we need to make sure there is the a `<s> [INST]` at the beginning of the first instruction.
            elif message:
                yield role + " ", message
            else:
                yield role, ""
        return
    if self.sep_style == SeparatorStyle.LLAMA3:
        if self.system_message:
            # For llama3, the system message is NOT incorporated into the first human instruction
            # All messages follow <|start_header_id|>' + role + '<|end_header_id|>\n\n'+ message + '<|eot_id|>
            yield "", system_prompt
        for i, (role, message) in enumerate(self.messages):
            if message:
                yield f"<|start_header_id|>{role}<|end_header_id|>\n\n", f"{message.strip()}<|eot_id|>"
            else:
                yield f"<|start_header_id|>{role}<|end_header_id|>\n\n", ""
        return
    if self.sep_style == SeparatorStyle.GEMMA:
        if self.system_message:
            raise ValueError("Gemma chat template does not support system messages")
        for i, (role, message) in enumerate(self.messages):
            prefix = "<bos>" if i == 0 else ""
            message_str = message if message else ""
            yield prefix + "<start_of_turn>" + role + "\n", message_str + "<end_of_turn>\n"
        return
    if self.sep_style == SeparatorStyle.CHATGLM:
        # source: https://huggingface.co/THUDM/chatglm-6b/blob/1d240ba371910e9282298d4592532d7f0f3e9f3e/modeling_chatglm.py#L1302-L1308
        # source2: https://huggingface.co/THUDM/chatglm2-6b/blob/e186c891cf64310ac66ef10a87e6635fa6c2a579/modeling_chatglm.py#L926
        round_add_n = 1 if self.name == "chatglm2" else 0
        if system_prompt:
            yield "", system_prompt + self.sep

        for i, (role, message) in enumerate(self.messages):
            if i % 2 == 0:
                yield "", f"[Round {i//2 + round_add_n}]{self.sep}"

            if message:
                yield f"{role}：", f"{message}{self.sep}"
            else:
                yield f"{role}：", ""
        return
    if self.sep_style == SeparatorStyle.CHATML:
        yield "", "" if system_prompt == "" else system_prompt + self.sep + "\n"
        for role, message in self.messages:
            if message:
                yield role + "\n", message + self.sep + "\n"
            else:
                yield role + "\n", ""
        return
    if self.sep_style == SeparatorStyle.CHATGLM3:
        if self.system_message:
            yield "", system_prompt
        for role, message in self.messages:
            if message:
                yield role + "\n", " " + message
            else:
                yield role
        return
    if self.sep_style == SeparatorStyle.CHATINTERN:
        # source: https://huggingface.co/internlm/internlm-chat-7b-8k/blob/bd546fa984b4b0b86958f56bf37f94aa75ab8831/modeling_internlm.py#L771
        seps = [self.sep, self.sep2]
        yield "", system_prompt
        for i, (role, message) in enumerate(self.messages):
            prefix = "<s>" if i % 2 == 0 else ""
            if message:
                yield prefix + role + ":", message + seps[i % 2] + "\n"
            else:
                yield role + ":", ""
        return
    if self.sep_style == SeparatorStyle.DOLLY:
        seps = [self.sep, self.sep2]
        yield "", system_prompt
        for i, (role, message) in enumerate(self.messages):
            if message:
                suffix = "\n\n" if i % 2 == 1 else ""
                yield role + ":\n", message + seps[i % 2] + suffix
            else:
                yield role + ":\n", ""
        return
    if self.sep_style == SeparatorStyle.PHOENIX:
        yield "", system_prompt
        for role, message in self.messages:
            if message:
                yield role + ": ", "<s>" + message + "</s>"
            else:
                yield role + ": " + "<s>", ""
        return
    if self.sep_style == SeparatorStyle.ROBIN:
        yield "", system_prompt + self.sep
        for role, message in self.messages:
            if message:
                yield role + ":\n", message + self.sep
            else:
                yield role + ":\n", ""
        return
    if self.sep_style == SeparatorStyle.FALCON_CHAT:
        if self.system_message:
            yield "", system_prompt + self.sep
        for role, message in self.messages:
            if message:
                yield role + ": ", message + self.sep
            else:
                yield role + ":", ""
    else:
        raise ValueError(f"Invalid style: {self.sep_style}")


def add_get_turns_to_conversation():
    import fastchat.conversation

    fastchat.conversation.Conversation.get_turns = get_turns
    fastchat.conversation.Conversation.get_prompt = get_prompt

def check_example_labels(example, tokenizer, text_only=False):
    # Get the input_ids, labels, and attention_mask from the dataset
    input_ids = example["input_ids"]
    labels = example["labels"]

    # You can compare the input_ids and labels element-wise
    # Remember to ignore positions with IGNORE_TOKEN_ID (if you use it) or attention_mask equal to 0
    colored_tokens = []
    for _, (input_id, label_id) in enumerate(zip(input_ids, labels)):
        decoded_input_token = tokenizer.decode(input_id)
        # Choose the color based on whether the label has the ignore value or not
        color = "red" if label_id == -100 else ("yellow" if label_id == 0 else "green")
        colored_token = colored(decoded_input_token, color) + (
            not text_only and colored(f"({label_id}, {input_id})", "white") or ""
        )
        colored_tokens.append(colored_token)

    delimiter = "" if text_only else " "
    return delimiter.join(colored_tokens)
