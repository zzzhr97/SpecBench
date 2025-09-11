import os
from rich.progress import Progress, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn, TextColumn
import concurrent.futures
import logging
import argparse

from specbench.utils import get_llm_engine, load_from_json, save_to_json, load_from_yaml, UsageRecorder
from specbench.ttd import get_ttd_engine

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/Child/prompts.json")
    parser.add_argument("--spec_path", type=str, default="data/Child/specifications.json")
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--ip", type=str, default="")
    parser.add_argument("--port", type=str, default="")
    parser.add_argument("--save_path", type=str, default="result/gpt-4.1-mini/Child/vanilla/generate.json")
    parser.add_argument("--adversarial", type=int, choices=[0, 1], default=1)
    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument("--method", type=str, default="vanilla")
    parser.add_argument("--ttd_config_path", type=str, default="config/config_ttd.yaml")
    parser.add_argument("--model_config_path", type=str, default="config/config_model.yaml")
    args = parser.parse_args()
    args.adversarial = bool(args.adversarial)
    for path in [args.data_path, args.spec_path, args.save_path]:
        if not path.endswith(".json"):
            raise ValueError(f"Path {path} is not ends with .json")
    return args

def wrap_inference(
    prompt,
    ttd_engine,
    **kwargs
):
    resp_dict = ttd_engine.inference(prompt)
    return resp_dict, kwargs

def generate_response(
    data_path, 
    spec_path,
    save_path,
    llm_engine,
    method="vanilla",
    adversarial=True,
    ttd_config_path=None,
    num_threads=128,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
    dataset = load_from_json(data_path)
    specification = load_from_json(spec_path)
    if adversarial:
        prompts = [data["adversarial_prompt"] for data in dataset]
    else:
        prompts = [data["prompt"] for data in dataset]
    
    ttd_engine = get_ttd_engine(
        llm_engine=llm_engine,
        method=method,
        specification=specification,
        ttd_config_path=ttd_config_path,
    )
    
    usage_recorder = UsageRecorder()
    total_response_dict = {idx: None for idx in range(len(prompts))}
    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(f"Generation | {method} | {llm_engine.model_name}", total=len(prompts))
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for idx, prompt in enumerate(prompts):
                future = executor.submit(
                    wrap_inference, 
                    prompt,
                    ttd_engine,
                    idx=idx,
                )
                futures.append(future)
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    resp_dict, kwargs = future.result()
                except Exception as e:
                    logging.error(f"Error in inference: {e}")
                    continue
                idx = kwargs["idx"]
                total_response_dict[idx] = {
                    "data": dataset[idx],
                    "prompt": prompts[idx],     # raw prompt without specification
                    "response": resp_dict["response"],
                    "thought": resp_dict["thought"],
                    "usage": resp_dict["usage"],
                }
                usage_recorder.update(resp_dict["usage"])
                cur_usage = usage_recorder.get_usage()
                prompt_tokens = cur_usage["prompt_tokens"]
                completion_tokens = cur_usage["completion_tokens"]
                total_tokens = cur_usage["total_tokens"]
                progress.update(task, advance=1, description=(
                    f"Generation | {method} | adv:{adversarial} | {llm_engine.model_name} | "
                    f"{prompt_tokens:09d} / {completion_tokens:09d} / {total_tokens:09d}"
                ))
                
    # re-sort the order
    total_response_list = []
    for idx in range(len(dataset)):
        total_response_list.append(total_response_dict[idx])
    
    # save generate.json
    save_to_json(data=total_response_list, json_path=save_path)
    
    # save usage.json
    usage_path = save_path.replace(".json", "_usage.json")
    save_to_json(data=usage_recorder.get_usage(), json_path=usage_path)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.ERROR,
        format="%(process)d-%(threadName)s-%(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()
    
    llm_engine = get_llm_engine(
        model=args.model,
        provider=args.provider,
        ip=args.ip,
        port=args.port,
        config_path=args.model_config_path,
    )
    
    total_responses = generate_response(
        data_path=os.path.expanduser(args.data_path), 
        spec_path=os.path.expanduser(args.spec_path), 
        save_path=os.path.expanduser(args.save_path),
        llm_engine=llm_engine,
        method=args.method,
        adversarial=args.adversarial,
        ttd_config_path=args.ttd_config_path,
        num_threads=args.num_threads,
    )