import os
import json
import warnings

from .prompt_ttd import (
    AugmentPrompt,
    SelfRefinePrompt,
    Align3Prompt,
    MoreThinkPrompt,
)
from .utils import UsageRecorder, load_from_yaml
    
CHAT_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "chat_templates")
def get_chat_template(model):
    with open(os.path.join(CHAT_TEMPLATE_DIR, "config.json"), "r") as f:
        chat_template_config = json.load(f)
    jinja_filename = chat_template_config[model]
    chat_template_path = os.path.join(CHAT_TEMPLATE_DIR, jinja_filename)
    if not os.path.exists(chat_template_path):
        chat_template = None
        print(f"Chat template not found: {chat_template_path}")
    else:
        with open(chat_template_path, "r") as f:
            chat_template = f.read()
        print(f"Use chat template from: {chat_template_path}")
    return chat_template

class TTDEngine:
    def __init__(self, 
        llm_engine, 
        method, 
        specification,
    ):
        self.llm_engine = llm_engine
        self.method = method
        self.specification = specification
        self.specification_string = self._format_specification_string()
        
    def _format_specification_string(self):
        safety_string = "".join([f"- {_}\n" for _ in self.specification["safety_specifications"]])
        behavioral_string = "".join([f"- {_}\n" for _ in self.specification["behavioral_specifications"]])
        specification_string = "**safety**\n" + safety_string + "**behavioral**\n" + behavioral_string
        return specification_string
        
    def _augment_prompt(self, prompt):
        augmented_prompt = AugmentPrompt.format(
            prompt=prompt,
            specification_string=self.specification_string,
        )
        return augmented_prompt
        
    def inference(self, prompt):
        augmented_prompt = self._augment_prompt(prompt)
        return self._inference_func(prompt=prompt, augmented_prompt=augmented_prompt)
    
    def _inference_func(self, prompt, augmented_prompt):
        raise NotImplementedError
    
class Raw(TTDEngine):
    def __init__(self, llm_engine, method, specification):
        super().__init__(llm_engine, method, specification)
        
    def _inference_func(self, prompt, augmented_prompt):
        usage_recorder = UsageRecorder()
        response_json = self.llm_engine.generate(prompt, usage_recorder=usage_recorder)
        thought = response_json["text"]["thought"]
        response = response_json["text"]["response"]
        return {
            "thought": thought,
            "response": response,
            "usage": usage_recorder.get_usage(),
        }
    
class Vanilla(TTDEngine):
    def __init__(self, llm_engine, method, specification):
        super().__init__(llm_engine, method, specification)
        
    def _inference_func(self, prompt, augmented_prompt):
        usage_recorder = UsageRecorder()
        response_json = self.llm_engine.generate(augmented_prompt, usage_recorder=usage_recorder)
        thought = response_json["text"]["thought"]
        response = response_json["text"]["response"]
        assert response != "", f"response is empty: {response_json}"

        return {
            "thought": thought,
            "response": response,
            "usage": usage_recorder.get_usage(),
        }
    
class SelfRefine(TTDEngine):
    def __init__(self, 
        llm_engine, 
        method, 
        specification,
        max_iters=15,
        max_history_length=1,
    ):
        super().__init__(llm_engine, method, specification)
        self.max_iters = max_iters
        self.max_history_length = max_history_length
        
    def _format_history_string(self, history):
        history_string = ""
        for idx, item in enumerate(history):
            history_string += f"<RESPONSE_{idx+1}>{item['response']}</RESPONSE_{idx+1}>"
            history_string += f"<FEEDBACK_{idx+1}>{item['feedback']}</FEEDBACK_{idx+1}>"
        return history_string
        
    def _inference_func(self, prompt, augmented_prompt):
        usage_recorder = UsageRecorder()
        
        response_json = self.llm_engine.generate(prompt=augmented_prompt, usage_recorder=usage_recorder)
        cur_response = response_json["text"]["response"]
        
        history = []
        for iter_idx in range(self.max_iters):
            # format feedback prompt
            feedback_system_prompt = SelfRefinePrompt.FeedBackSystem.format(
                query=prompt,
                specification_string=self.specification_string,
            )
            feedback_user_prompt = SelfRefinePrompt.FeedBackUser.format(
                response=cur_response,
            )
            feedback_messages = [
                {"role": "system", "content": feedback_system_prompt},
                {"role": "user", "content": feedback_user_prompt},
            ]
            
            # generate feedback
            response_json = self.llm_engine.generate(messages=feedback_messages, usage_recorder=usage_recorder)
            feedback = response_json["text"]["response"]
            
            # update history
            history.append({
                "response": cur_response,
                "feedback": feedback,
            })
            
            # format history string (only keep the last max_history_length items)
            history_string = self._format_history_string(history[-self.max_history_length:])
            
            # format refine prompt
            refine_system_prompt = SelfRefinePrompt.RefineSystem.format(
                query=prompt,
            )
            refine_user_prompt = SelfRefinePrompt.RefineUser.format(
                history_string=history_string,
            )
            refine_messages = [
                {"role": "system", "content": refine_system_prompt},
                {"role": "user", "content": refine_user_prompt},
            ]
            
            # generate refined response
            response_json = self.llm_engine.generate(messages=refine_messages, usage_recorder=usage_recorder)
            cur_response = response_json["text"]["response"]
            
        return {
            "response": cur_response,
            "thought": "",
            "usage": usage_recorder.get_usage(),
        }
        
class Align3(TTDEngine):
    def __init__(self, llm_engine, method, specification):
        super().__init__(llm_engine, method, specification)
        self.chat_template = get_chat_template(llm_engine.model_name)
        
        self.safety_string = "".join([f"- {_}\n" for _ in self.specification["safety_specifications"]])
        self.behavioral_string = "".join([f"- {_}\n" for _ in self.specification["behavioral_specifications"]])
        
        self.extra_config_stage_1_2 = {
            "stop": ["</think>"],
            "extra_body": {
                "add_generation_prompt": False,
                "continue_final_message": True,
                "chat_template": self.chat_template,
                "chat_template_kwargs": {
                    "enable_thinking": False,
                    "enable_raw_content": True,
                }
            },
        }
        self.extra_config_stage_3 = {
            "stop": [],
            "extra_body": {
                "add_generation_prompt": False,
                "continue_final_message": True,
                "chat_template": self.chat_template,
                "chat_template_kwargs": {
                    "enable_thinking": False,
                    "enable_raw_content": True,
                }
            },
        }
        
    def _inference_func(self, prompt, augmented_prompt):
        raw_messages = self.llm_engine._format_messages(augmented_prompt)
        usage_recorder = UsageRecorder()
        
        # First stage
        first_stage_thought_prompt = Align3Prompt.FirstStage.format(
                behavioral_specifications=self.behavioral_string,
        )
        messages = [
            *raw_messages,
            {"role": "assistant", "content": first_stage_thought_prompt}
        ]
        response_json = self.llm_engine.generate(messages=messages, extra_config=self.extra_config_stage_1_2, usage_recorder=usage_recorder)
        prompt_tokens = usage_recorder.get_usage("prompt_tokens")
        first_stage_response = response_json["text"]["total"]
        
        # Second stage
        second_stage_thought_prompt = first_stage_thought_prompt + first_stage_response.rstrip() + "\n\n" + Align3Prompt.SecondStage.format(
            safety_specifications=self.safety_string,
        )
        messages = [
            *raw_messages,
            {"role": "assistant", "content": second_stage_thought_prompt}
        ]
        response_json = self.llm_engine.generate(messages=messages, extra_config=self.extra_config_stage_1_2, usage_recorder=usage_recorder)
        second_stage_response = response_json["text"]["total"]
        
        # Third stage
        third_stage_thought_prompt = second_stage_thought_prompt + second_stage_response.rstrip() + "\n\n" + Align3Prompt.ThirdStage.format(
            safety_specifications=self.safety_string,
            behavioral_specifications=self.behavioral_string,
        )
        messages = [
            *raw_messages,
            {"role": "assistant", "content": third_stage_thought_prompt}
        ]
        response_json = self.llm_engine.generate(messages=messages, extra_config=self.extra_config_stage_3, usage_recorder=usage_recorder)
        third_stage_thought = response_json["text"]["thought"]
        third_stage_response = response_json["text"]["response"]
        
        usage_recorder.set_usage("prompt_tokens", prompt_tokens)
        
        # parse thought and response
        thought = third_stage_thought_prompt + third_stage_thought
        response = third_stage_response
        
        return {
            "response": response,
            "thought": thought,
            "usage": usage_recorder.get_usage(),
        }
        
class MoreThink(TTDEngine):
    def __init__(self, llm_engine, method, specification, rethink_count=3):
        super().__init__(llm_engine, method, specification)
        self.rethink_count = rethink_count
        self.chat_template = get_chat_template(llm_engine.model_name)
        
        self.extra_config_thinking = {
            "stop": ["</think>"],
            "extra_body": {
                "include_stop_str_in_output": False,
                "add_generation_prompt": False,
                "continue_final_message": True,
                "chat_template": self.chat_template,
                "chat_template_kwargs": {
                    "enable_thinking": False,
                    "enable_raw_content": True,
                },
            },
        }
        self.extra_config_response = {
            "stop": [],
            "extra_body": {
                "add_generation_prompt": False,
                "continue_final_message": True,
                "chat_template": self.chat_template,
                "chat_template_kwargs": {
                    "enable_thinking": False,
                    "enable_raw_content": True,
                }
            },
        }
        
    def _inference_func(self, prompt, augmented_prompt):
        raw_messages = self.llm_engine._format_messages(augmented_prompt)
        usage_recorder = UsageRecorder()
        
        history_string = "<think>"
        for i in range(self.rethink_count - 1):
            messages = [
                *raw_messages,
                {"role": "assistant", "content": history_string}
            ]
            response_json = self.llm_engine.generate(messages=messages, extra_config=self.extra_config_thinking, usage_recorder=usage_recorder)
            
            if i == 0:
                prompt_tokens = usage_recorder.get_usage("prompt_tokens")
            
            response = response_json["text"]["total"]
            history_string += response.rstrip() + "\n\n" + MoreThinkPrompt.Rethink
        
        messages = [
            *raw_messages,
            {"role": "assistant", "content": history_string}
        ]
        response_json = self.llm_engine.generate(messages=messages, extra_config=self.extra_config_response, usage_recorder=usage_recorder)
        thought = history_string + response_json["text"]["thought"]
        response = response_json["text"]["response"]
        usage_recorder.set_usage("prompt_tokens", prompt_tokens)
        
        return {
            "response": response,
            "thought": thought,
            "usage": usage_recorder.get_usage(),
        }
        
class ZeroThink(TTDEngine):
    def __init__(self, llm_engine, method, specification):
        super().__init__(llm_engine, method, specification)
        self.chat_template = get_chat_template(llm_engine.model_name)
        
        self.extra_config_thinking = {
            "stop": ["</think>"],
            "extra_body": {
                "include_stop_str_in_output": False,
                "add_generation_prompt": False,
                "continue_final_message": True,
                "chat_template": self.chat_template,
                "chat_template_kwargs": {
                    "enable_thinking": False,
                    "enable_raw_content": True,
                },
            },
        }
        self.extra_config_response = {
            "stop": [],
            "extra_body": {
                "add_generation_prompt": False,
                "continue_final_message": True,
                "chat_template": self.chat_template,
                "chat_template_kwargs": {
                    "enable_thinking": False,
                    "enable_raw_content": True,
                }
            },
        }
        
    def _inference_func(self, prompt, augmented_prompt):
        raw_messages = self.llm_engine._format_messages(augmented_prompt)
        usage_recorder = UsageRecorder()
        zero_think_prefix = "<think>\n</think>"
        messages = [
            *raw_messages,
            {"role": "assistant", "content": zero_think_prefix}
        ]
        response_json = self.llm_engine.generate(messages=messages, extra_config=self.extra_config_response, usage_recorder=usage_recorder)
        thought = zero_think_prefix + response_json["text"]["thought"]
        response = response_json["text"]["response"]
        
        return {
            "response": response,
            "thought": thought,
            "usage": usage_recorder.get_usage(),
        }
        
def get_ttd_engine(
    llm_engine, 
    method, 
    specification,
    ttd_config_path=None,
    ttd_config=None,
):      
    if ttd_config is not None:
        if ttd_config_path is not None:
            warnings.warn(
                f"Both `ttd_config` and `ttd_config_path` ({ttd_config_path}) are provided. "
                "The explicit `ttd_config` dictionary will take precedence."
            )
    elif ttd_config_path:
        if os.path.exists(ttd_config_path):
            all_ttd_config = load_from_yaml(ttd_config_path)
            ttd_config = all_ttd_config.get(method, {})
        else:
            warnings.warn(
                f"The provided ttd_config_path \"{ttd_config_path}\" does not exist. "
                "Falling back to default ttd_config."
            )
            ttd_config = {}
    else:
        warnings.warn(
            "Neither `ttd_config` nor `ttd_config_path` is provided. "
            "Using default ttd_config."
        )
        ttd_config = {}
    
    print(f"Method: {method}, ttd_config: {ttd_config}")
    if method == "raw":
        return Raw(llm_engine, method, specification)
    elif method == "vanilla":
        return Vanilla(llm_engine, method, specification)
    elif method == "self_refine":
        max_iters = ttd_config.get("max_iters", 3)
        max_history_length = ttd_config.get("max_history_length", 1)
        return SelfRefine(llm_engine, method, specification, max_iters, max_history_length)
    elif method == "align3":
        return Align3(llm_engine, method, specification)
    elif method == "more_think":
        rethink_count = ttd_config.get("rethink_count", 3)
        return MoreThink(llm_engine, method, specification, rethink_count)
    elif method == "zero_think":
        return ZeroThink(llm_engine, method, specification)
    else:
        raise ValueError(f"Invalid TTD method: {method}")