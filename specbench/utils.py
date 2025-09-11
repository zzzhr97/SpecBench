import backoff
from openai import OpenAI, RateLimitError
from rich.progress import Progress, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn, TextColumn
import concurrent.futures
import os
import json
import yaml
import logging
import threading
import warnings
from abc import ABC
import copy

def load_from_json(json_path):
    with open(json_path, "r") as f:
        return json.load(f)
def save_to_json(data, json_path):
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
def load_from_yaml(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

class Metric:
    def __init__(self, description):
        self.description = description
        self.reset()
        
    def reset(self):
        self.total_value = 0
        self.count = 0
        self.value_list = []
        
    def add(self, value):
        self.total_value += value
        self.count += 1
        self.value_list.append(value)

    def avg_score(self):
        if self.count == 0:
            return 0
        else:
            return self.total_value / self.count
        
    def repr_avg_string(self):
        return f"{self.avg_score():.4f} ({self.total_value:.4f} / {self.count})"
    
    def repr_avg_string_raw(self):
        return f"{self.avg_score():.4f}"

class UsageRecorder:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()
    def update(self, usage):
        with self.lock:
            self.usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
            self.usage["completion_tokens"] += usage.get("completion_tokens", 0)
            self.usage["total_tokens"] += usage.get("total_tokens", 0)
    def reset(self):
        with self.lock:
            self.usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
    def get_usage(self, key=None):
        with self.lock:
            if key is None:
                return self.usage
            else:
                return self.usage[key]
    def set_usage(self, key, value):
        with self.lock:
            assert key in ["prompt_tokens", "completion_tokens"]
            self.usage[key] = value
            self.usage["total_tokens"] = self.usage["prompt_tokens"] + self.usage["completion_tokens"]

def deep_update(d, u):
    nd = copy.deepcopy(d)
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(nd.get(k), dict):
            nd[k] = deep_update(nd[k], v)
        else:
            nd[k] = copy.deepcopy(v)
    return nd

class LLMEngine(ABC):
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."
    def __init__(
        self,
        provider="vllm-server",
        model="Qwen/Qwen3-32B-thinking",
        config={},
        system_prompt=None,
        **kwargs,
    ):
        self.provider = provider
        self.model = model
        self.model_name = config.pop("model_name", model)
        self.config = config
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.lock = threading.Lock()
        print(f"Model: {self.model}, model_name: {self.model_name}, config: {self.config}")
        
    def _format_messages(self, prompt):
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
    def _format_response_json(self, response):
        finish_reason = response.choices[0].finish_reason
        if finish_reason in ['content_filter: PROHIBITED_CONTENT']:
            total_response_text = "I'm sorry, but I can't assist with that request."
        else:
            total_response_text = response.choices[0].message.content
            
        if total_response_text is None:
            logging.error(f"None error: {response}")
        
        # thu-ml/STAIR-Llama-3.1-8B-DPO-3
        if self.model_name == "thu-ml/STAIR-Llama-3.1-8B-DPO-3":
            thought_text, response_text = total_response_text.split("Final Answer: ")
        # other normal models and RealSafe/RealSafe-R1-8B
        else:
            if self.model_name.startswith("gemini"):
                end_of_thinking = "</thought>"
            else:
                end_of_thinking = "</think>"
            response_text = total_response_text.split(end_of_thinking)[-1].lstrip('\n')
            try:
                thought_text = total_response_text.split(end_of_thinking)[-2].split(end_of_thinking)[-1].rstrip('\n')
            except IndexError:
                thought_text = ""
            except Exception as e:
                raise e
        response_json = {
            "text": {
                "response": response_text,
                "thought": thought_text,
                "total": total_response_text,
            },
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        }
        return response_json
        
    def generate(self, prompt=None, messages=None, extra_config={}, usage_recorder=None, **kwargs):
        if (prompt is None) == (messages is None):
            raise ValueError("Either prompt or messages must be provided, but not both")
        if prompt is not None:
            messages = self._format_messages(prompt)
        config = deep_update(self.config, extra_config)
        response_json = self._generate(messages, config=config, **kwargs)
        if usage_recorder is not None:
            usage_recorder.update(response_json["usage"])
        return response_json
    
    def batch_generate(self, prompts=None, messages_list=None, extra_config={}, usage_recorder=None, **kwargs):
        if (prompts is None) == (messages_list is None):
            raise ValueError("Either prompts or messages_list must be provided, but not both")
        if prompts is not None:
            messages_list = [self._format_messages(prompt) for prompt in prompts]
        config = deep_update(self.config, extra_config)
        response_json_list = self._batch_generate(messages_list, config=config, **kwargs)
        if usage_recorder is not None:
            for response_json in response_json_list:
                usage_recorder.update(response_json["usage"])
        return response_json_list
    
    def _generate(self, messages, config={}, **kwargs):
        raise NotImplementedError
    
    def _batch_generate(self, messages_list, config={}, **kwargs):
        raise NotImplementedError
    
    def __call__(self, prompt):
        return self.generate(prompt)
    
class OpenAIEngine(LLMEngine):
    PROVIDER = "openai"
    BASE_URL = "https://api.openai.com/v1"
    API_KEY_STRING = "OPENAI_API_KEY"
    def __init__(self,
        model="gpt-4.1",
        config={},
        system_prompt=None,
        **kwargs,
    ):
        super().__init__(
            provider=self.PROVIDER,
            model=model,
            config=config,
            system_prompt=system_prompt,
            **kwargs,
        )
        self.client = OpenAI(api_key=os.getenv(self.API_KEY_STRING), base_url=self.BASE_URL)
        
    @backoff.on_exception(backoff.expo, RateLimitError)
    def _generate(self, messages, config={}, **kwargs):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **config
        )
        return self._format_response_json(response)
    
    def _batch_generate(self, *args, **kwargs):
        raise NotImplementedError
        
class DeepSeekAIEngine(LLMEngine):
    PROVIDER = "deepseek-ai"
    BASE_URL = "https://api.deepseek.com"
    API_KEY_STRING = "DEEPSEEK_API_KEY"
    def __init__(self,
        model="deepseek-chat",
        config={},
        system_prompt=None,
        **kwargs,
    ):
        super().__init__(
            provider=self.PROVIDER,
            model=model,
            config=config,
            system_prompt=system_prompt,
            **kwargs,
        )
        self.client = OpenAI(api_key=os.getenv(self.API_KEY_STRING), base_url=self.BASE_URL)
        
    def _format_response_json_deepseek(self, response):
        response_text = response.choices[0].message.content
        try:
            thought_text = response.choices[0].message.reasoning_content
            if type(thought_text) != str:
                thought_text = ""
        except Exception as e:
            thought_text = ""
        total_response_text = thought_text + "\n</think>\n" + response_text
        response_json = {
            "text": {
                "response": response_text,
                "thought": thought_text,
                "total": total_response_text
            },
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        }
        return response_json
        
    @backoff.on_exception(backoff.expo, RateLimitError)
    def _generate(self, messages, config={}, **kwargs):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **config
        )
        return self._format_response_json_deepseek(response)
    
    def _batch_generate(self, *args, **kwargs):
        raise NotImplementedError
        
class VllmEngine(LLMEngine):
    PROVIDER = "vllm-server"
    def __init__(self,
        model="Qwen/Qwen3-32B-thinking",
        config={},
        system_prompt=None,
        ip="localhost",
        port="8001",
        num_threads=16,
        **kwargs,
    ):
        super().__init__(
            provider=self.PROVIDER,
            model=model,
            config=config,
            system_prompt=system_prompt,
            **kwargs,
        )
        self.ip = ip
        self.port = port
        self.num_threads = num_threads
        self._init_vllm_engine()

    def _init_vllm_engine(self):
        self.lock = threading.Lock()
        ips = self.ip.split(",")
        ports = self.port.split(",")
        # os.environ["NO_PROXY"] = self.ip
        
        self.vllm_engines = []
        for ip, port in zip(ips, ports):
            base_url = f"http://{ip}:{port}/v1"
            print(f"vllm server | {self.model_name}: {base_url}")
            self.vllm_engines.append(OpenAI(
                base_url=base_url,
                api_key="token-abc123",
            ))
        self.total = len(self.vllm_engines)
        self.engine_idx = 0
        
    def _get_cur_engine_idx(self):
        with self.lock:
            self.engine_idx = (self.engine_idx + 1) % self.total
            engine_idx = self.engine_idx
        return engine_idx
        
    def _generate(self, messages, config={}, **kwargs):
        engine_idx = self._get_cur_engine_idx()
        try:
            stream = config.pop("stream", False)
            n = config.get("n", 1)
            
            if not stream:
                response = self.vllm_engines[engine_idx].chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    **config
                )
                
                if n > 1:
                    return [self._format_response_json(response[i]) for i in range(n)]
                else:
                    return self._format_response_json(response)
            else:
                if n > 1:
                    raise NotImplementedError("Stream mode does not support multiple responses")    
                
                response = self.vllm_engines[engine_idx].chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    stream=True,
                    **config
                )
                # reasoning content or content
                return response
        except Exception as e:
            logging.error(f"Engine {engine_idx} error: {e}")
            return self._generate(messages, config=config, **kwargs)
    
    def _warp_generate(self, messages, config={}, **kwargs):
        return self._generate(messages, config=config), kwargs
    
    def _batch_generate(self, messages_list, num_threads=None, config={}, **kwargs):
        num_threads = num_threads or self.num_threads
        response_dict = {idx: None for idx in range(len(messages_list))}
        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(f"Batch Generation | {num_threads} threads", total=len(messages_list))
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = []
                for idx, messages in enumerate(messages_list):
                    future = executor.submit(
                        self._warp_generate, 
                        messages,
                        config=config,
                        idx=idx,
                    )
                    futures.append(future)
                
                for future in concurrent.futures.as_completed(futures):
                    response, _kwargs = future.result()
                    idx = _kwargs["idx"]
                    response_dict[idx] = response
                    progress.update(task, advance=1)
                    
        response_list = [response_dict[idx] for idx in range(len(messages_list))]
        return response_list
    
def get_llm_engine(
    provider="vllm-server",
    model="Qwen/Qwen3-32B-thinking",
    config_path=None,
    config=None,
    **kwargs,
):
    if config is not None:
        if config_path is not None:
            warnings.warn(
                f"Both `config` and `config_path` ({config_path}) are provided. "
                "The explicit `config` dictionary will take precedence."
            )
    elif config_path:
        if os.path.exists(config_path):
            all_config = load_from_yaml(config_path)
            config = all_config.get(model, {})
        else:
            warnings.warn(
                f"The provided config_path \"{config_path}\" does not exist. "
                "Falling back to an empty config."
            )
            config = {}
    else:
        warnings.warn(
            "Neither `config` nor `config_path` is provided. "
            "Using an empty config."
        )
        config = {}
        
    if provider == "openai":
        return OpenAIEngine(model=model, config=config, **kwargs)
    elif provider == "vllm-server":
        return VllmEngine(model=model, config=config, **kwargs)
    elif provider == "deepseek-ai":
        return DeepSeekAIEngine(model=model, config=config, **kwargs)
    else:
        raise ValueError(f"Invalid provider: {provider}")