
import os
from rich.progress import Progress, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn, TextColumn
import concurrent.futures
import json
import logging
import argparse

from specbench.evaluator import JointEvaluator, SequentialEvaluator
from specbench.utils import get_llm_engine, Metric, load_from_json, UsageRecorder

RESPONSE_MAX_LENGTH = 8000

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--response_data_path", type=str, default="result/gpt-4.1-mini/Child/vanilla/generate.json")
    parser.add_argument("--eval_model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--eval_model_provider", type=str, default="openai")
    parser.add_argument("--eval_save_path", type=str, default="result/gpt-4.1-mini/Child/vanilla/gpt-4.1-mini_evaluate.json")
    parser.add_argument("--score_save_path", type=str, default="result/gpt-4.1-mini/Child/vanilla/gpt-4.1-mini_score.json")
    parser.add_argument("--spec_path", type=str, default="data/Child/specifications.json")
    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument("--ip", type=str, default="localhost")
    parser.add_argument("--port", type=str, default="8001")
    parser.add_argument("--eval_config_path", type=str, default="config/config_eval_model.yaml")
    parser.add_argument("--evaluation_type", type=str, choices=["joint", "sequential"], default="joint")
    args = parser.parse_args()
    return args

def wrap_evaluate(
    response_data,
    evaluator,
    **kwargs
):
    try:
        data, response = response_data["data"], response_data["response"]
        response = response[:RESPONSE_MAX_LENGTH]
        eval_result = evaluator.evaluate([data], [response])[0]
        return eval_result, kwargs
    except Exception as e:
        logging.error(f"Error evaluating response: {e}")
        logging.error(f"Response Data: {response_data}")
        return None, kwargs

def print_and_save_result(
    total_result,
    metrics,
    eval_save_path,
    score_save_path
):
    print("| =========== unsafe subset ========== | ============ safe subset =========== | \033[33m============ full dataset ==========\033[0m |")
    for metric_type in ["safety", "behavioral", "SAR"]:
        print("| ", end="")
        for data_type in ["unsafe_subset", "safe_subset", "full_dataset"]:
            if data_type == "full_dataset":
                print("\033[33m", end="")
            print(f"{metric_type:<10}: {metrics[data_type][metric_type].repr_avg_string():<24}", end="")
            if data_type == "full_dataset":
                print("\033[0m", end="")
            print(" | ", end="")
        print()
    
    if eval_save_path:
        with open(eval_save_path, "w") as f:
            json.dump(total_result, f, indent=4, ensure_ascii=False)
    if score_save_path:
        score_result = {}
        for metric in [
            metrics["full_dataset"]["safety"],
            metrics["full_dataset"]["behavioral"], 
            metrics["full_dataset"]["SAR"],
            metrics["unsafe_subset"]["safety"],
            metrics["unsafe_subset"]["behavioral"],
            metrics["unsafe_subset"]["SAR"],
            metrics["safe_subset"]["safety"],
            metrics["safe_subset"]["behavioral"],
            metrics["safe_subset"]["SAR"],
        ]:
            score_result[metric.description] = {
                "avg": metric.avg_score(),
                "total": metric.total_value,
                "count": metric.count,
            }
        with open(score_save_path, "w") as f:
            json.dump(score_result, f, indent=4, ensure_ascii=False)

def evaluate_response(
    data_path, 
    spec_path,
    eval_save_path,
    score_save_path,
    llm_engine,
    evaluation_type="joint",
    num_threads=8
):
    if eval_save_path:
        os.makedirs(os.path.dirname(eval_save_path), exist_ok=True)
    dataset = load_from_json(data_path)
    specification = load_from_json(spec_path)
    if evaluation_type == "joint":
        evaluator = JointEvaluator(llm_engine, specification)
    if evaluation_type == "sequential":
        evaluator = SequentialEvaluator(llm_engine, specification)
    
    total_result_dict = {}
    metrics = {}
    for data_type in ["unsafe_subset", "safe_subset", "full_dataset"]:
        metrics[data_type] = {
            "safety": Metric(f"{data_type}_safety"),
            "behavioral": Metric(f"{data_type}_behavioral"),
            "SAR": Metric(f"{data_type}_SAR"),
        }
        
    usage_recorder = UsageRecorder()
    
    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(f"Evaluating Response | {num_threads} threads", total=len(dataset))
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for idx, data in enumerate(dataset):
                future = executor.submit(
                    wrap_evaluate, 
                    data,
                    evaluator,
                    idx=idx,
                )
                futures.append(future)
            
            for future in concurrent.futures.as_completed(futures):
                eval_result, kwargs = future.result()
                idx = kwargs["idx"]
                usage_recorder.update(eval_result["usage"])
                cur_total_result = {
                    "data": dataset[idx]["data"],
                    "prompt": dataset[idx]["prompt"],   # raw prompt without specification
                    "response": dataset[idx]["response"],
                    "thought": dataset[idx].get("thought", None),
                    "eval": eval_result,
                }
                total_result_dict[idx] = cur_total_result
                
                safety_score = eval_result["result"]["score"]["safety"]
                behavioral_score = eval_result["result"]["score"]["behavioral"]
                total_score = eval_result["result"]["score"]["total"]
                cur_data_type = dataset[idx]["data"]["label"]
                for data_type_record in ["full_dataset", cur_data_type + "_subset"]:
                    metrics[data_type_record]["safety"].add(safety_score)
                    metrics[data_type_record]["behavioral"].add(behavioral_score)
                    metrics[data_type_record]["SAR"].add(total_score)
                    
                progress.update(
                    task, 
                    advance=1,
                    description=(
                        f"[Unsafe] S: {metrics['unsafe_subset']['safety'].repr_avg_string_raw()} | "
                        f"B: {metrics['unsafe_subset']['behavioral'].repr_avg_string_raw()} | "
                        f"T: {metrics['unsafe_subset']['SAR'].repr_avg_string_raw()} "
                        f"---- [Safe] S: {metrics['safe_subset']['safety'].repr_avg_string_raw()} | "
                        f"B: {metrics['safe_subset']['behavioral'].repr_avg_string_raw()} | "
                        f"T: {metrics['safe_subset']['SAR'].repr_avg_string_raw()} "
                        "\033[33m"
                        f"---- [Full] S: {metrics['full_dataset']['safety'].repr_avg_string_raw()} | "
                        f"B: {metrics['full_dataset']['behavioral'].repr_avg_string_raw()} | "
                        f"T: {metrics['full_dataset']['SAR'].repr_avg_string_raw()} "
                        f"\033[0m"
                ))
    
    # re-sort the order
    total_result = []
    for idx in range(len(dataset)):
        total_result.append(total_result_dict[idx])
    
    print_and_save_result(
        total_result,
        metrics,
        eval_save_path,
        score_save_path,
    )

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.ERROR,
        format="%(process)d-%(threadName)s-%(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()
    
    llm_engine = get_llm_engine(
        model=args.eval_model,
        provider=args.eval_model_provider,
        ip=args.ip,
        port=args.port,
        config_path=args.eval_config_path
    )
    
    total_result = evaluate_response(
        data_path=os.path.expanduser(args.response_data_path), 
        spec_path=args.spec_path,
        eval_save_path=os.path.expanduser(args.eval_save_path),
        score_save_path=os.path.expanduser(args.score_save_path),
        llm_engine=llm_engine,
        evaluation_type=args.evaluation_type,
        num_threads=args.num_threads,
    )