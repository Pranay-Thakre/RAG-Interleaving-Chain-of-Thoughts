import requests
import os
from typing import Dict

from diskcache import Cache
from commaqa.inference.prompt_reader import fit_prompt_into_given_limit

cache = Cache(os.path.expanduser("~/.cache/llmcalls"))


def non_cached_llm_call(  # kwargs doesn't work with caching.
    prompt,
    model_name,
    max_input=None,
    max_length=100,
    min_length=1,
    do_sample=False,
    temperature=1.0,
    top_k=50,
    top_p=1.0,
    num_return_sequences=1,
    repetition_penalty=None,
    length_penalty=None,
    keep_prompt=False,
) -> Dict:

    params = {
        "prompt": prompt,
        "max_input": max_input,
        "max_length": max_length,
        "min_length": min_length,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "num_return_sequences": num_return_sequences,
        "repetition_penalty": repetition_penalty,
        "length_penalty": length_penalty,
        "keep_prompt": keep_prompt,
    }

    if model_name == "mock" or os.environ.get("USE_MOCK_LLM") == "1":
        from commaqa.models.mock_generator import MockGenerator
        mg = MockGenerator(engine=model_name)
        seqs = mg.generate_text_sequence(prompt)
        return {
            "model_name": model_name,
            "prompt": prompt,
            "output_seq_score": seqs
        }

    # Check for Ollama server or custom free LLM server
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    if os.environ.get("USE_OLLAMA") == "1":
        try:
            ollama_model = os.environ.get("OLLAMA_MODEL", "llama3")
            payload = {
                "model": ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature}
            }
            res = requests.post(f"{ollama_host}/api/generate", json=payload, timeout=30)
            if res.status_code == 200:
                gen_text = res.json().get("response", "")
                return {"generated_texts": [gen_text], "model_name": ollama_model}
        except Exception as e:
            print(f"[LLMClientGenerator] Ollama call failed ({e}), falling back to standard LLM server / Mock.")

    host = os.environ.get("LLM_SERVER_HOST", "http://localhost")
    port = os.environ.get("LLM_SERVER_PORT", "8010")

    if "/" in model_name:
        assert model_name.count("/", 1)
        model_name = model_name.split("/")[1]

    llm_server_key_suffix = os.environ.get("LLM_SERVER_KEY_SUFFIX", "")
    if model_name.replace("-", "_") + "_LLM_SERVER_HOST" in os.environ:
        host = os.environ[model_name.replace("-", "_") + "_LLM_SERVER_HOST" + llm_server_key_suffix]
    if model_name.replace("-", "_") + "_LLM_SERVER_PORT" in os.environ:
        port = os.environ[model_name.replace("-", "_") + "_LLM_SERVER_PORT" + llm_server_key_suffix]

    try:
        url = f"{host.rstrip('/')}:{port}/generate" if ":" not in host.split("//")[-1] else f"{host.rstrip('/')}/generate"
        response = requests.get(url, params=params, timeout=30)
        if response.status_code != 200:
            raise Exception("LLM Generation request failed!")
        result = response.json()
        return result
    except Exception as e:
        print(f"[LLMClientGenerator] LLM server request failed ({e}), falling back to MockGenerator.")
        from commaqa.models.mock_generator import MockGenerator
        mg = MockGenerator(engine=model_name)
        seqs = mg.generate_text_sequence(prompt)
        return {
            "model_name": model_name,
            "prompt": prompt,
            "output_seq_score": seqs
        }




@cache.memoize()
def cached_llm_call(  # kwargs doesn't work with caching.
    prompt,
    model_name,
    max_input=None,
    max_length=100,
    min_length=1,
    do_sample=False,
    temperature=1.0,
    top_k=50,
    top_p=1.0,
    num_return_sequences=1,
    repetition_penalty=None,
    length_penalty=None,
    keep_prompt=False,
) -> Dict:
    return non_cached_llm_call(
        prompt,
        model_name,
        max_input=max_input,
        max_length=max_length,
        min_length=min_length,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=num_return_sequences,
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
        keep_prompt=keep_prompt,
    )


def llm_call(
    prompt,
    model_name,
    max_input=None,
    max_length=100,
    min_length=1,
    do_sample=False,
    temperature=1.0,
    top_k=50,
    top_p=1.0,
    num_return_sequences=1,
    repetition_penalty=None,
    length_penalty=None,
    keep_prompt=False,
) -> Dict:
    function = cached_llm_call if not do_sample and temperature > 0 else non_cached_llm_call
    return function(
        prompt,
        model_name,
        max_input=max_input,
        max_length=max_length,
        min_length=min_length,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=num_return_sequences,
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
        keep_prompt=keep_prompt,
    )


class LLMClientGenerator:

    # Instructions to start the LLM Server are in the README here:
    # https://github.com/harshTrivedi/llm_server

    def __init__(
        self,
        model_name,
        max_input=None,
        max_length=100,
        min_length=1,
        do_sample=False,
        eos_text="\n",
        temperature=1.0,
        top_k=50,
        top_p=1.0,
        num_return_sequences=1,
        repetition_penalty=None,
        length_penalty=None,
        model_tokens_limit=2000,
        remove_method="first",
    ):

        valid_model_names = [
            "gpt-j-6B",
            "opt-66b",
            "gpt-neox-20b",
            "T0pp",
            "flan-t5-base",
            "flan-t5-large",
            "flan-t5-xl",
            "flan-t5-xxl",
            "ul2",
        ]
        model_name_ = model_name
        if "/" in model_name:
            assert model_name.count("/", 1)
            model_name_ = model_name.split("/")[1]
        assert model_name_ in valid_model_names, f"Model name {model_name_} not in {valid_model_names}"

        self.model_name = model_name
        self.max_input = max_input
        self.max_length = max_length
        self.min_length = min_length
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.eos_text = eos_text
        self.num_return_sequences = num_return_sequences
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty
        self.model_tokens_limit = model_tokens_limit
        self.remove_method = remove_method

    def generate_text_sequence(self, prompt):
        """
        :param input_text:
        :return: returns a sequence of tuples (string, score) where lower score is better
        """
        prompt = prompt.rstrip()

        prompt = fit_prompt_into_given_limit(
            original_prompt=prompt,
            model_length_limit=self.model_tokens_limit,
            estimated_generation_length=self.max_length,
            demonstration_delimiter="\n\n\n",
            shuffle=False,
            remove_method=self.remove_method,
            tokenizer_model_name=self.model_name,
            last_is_test_example=True,
        )

        # Note: Don't pass eos_text. Doesn't seem to work right.
        params = {
            "prompt": prompt,
            "model_name": self.model_name,
            "max_input": self.max_input,
            "max_length": self.max_length,
            "min_length": self.min_length,
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "num_return_sequences": self.num_return_sequences,
            "repetition_penalty": self.repetition_penalty,
            "length_penalty": self.length_penalty,
            "keep_prompt": False,
        }
        result = llm_call(**params)

        generated_texts = result.get("generated_texts")
        if generated_texts is None:
            generated_texts = [x[0] for x in result.get("output_seq_score", [])]

        modified_texts = []
        for text in generated_texts:
            # remove the prompt
            if text.startswith(prompt):
                text = text[len(prompt) :]
            if self.eos_text and self.eos_text in text:
                text = text[: text.index(self.eos_text)]
            modified_texts.append(text)
        generated_texts = modified_texts

        output_seq_score = [(text, 1 / (index + 1)) for index, text in enumerate(generated_texts)]
        # print(prompt)
        # print("------------")
        # print(output_seq_score[0][0])

        # TODO: Deal with output-probabilities if needed.

        return sorted(output_seq_score, key=lambda x: x[1])
