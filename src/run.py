import os
import argparse
import logging

from specbench.utils import get_llm_engine
from generate_response import generate_response
from evaluate_response import evaluate_response

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="Child")
    parser.add_argument("--mode", type=str, choices=["generation", "evaluation"], default="generation")
    
    parser.add_argument("--model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--eval_model", type=str, default="gpt-4.1")
    parser.add_argument("--eval_model_provider", type=str, default="openai")
    
    parser.add_argument("--evaluation_type", type=str, choices=["joint", "sequential"], default="joint")
    
    parser.add_argument("--method", type=str, default="vanilla")
    parser.add_argument("--num_threads", type=int, default=8)
    
    parser.add_argument("--ip", type=str, default="localhost")
    parser.add_argument("--port", type=str, default="8001")
    
    parser.add_argument("--adversarial", type=int, choices=[0, 1], default=1)
    args = parser.parse_args()
    args.adversarial = bool(args.adversarial)
    
    model_without_org = args.model.split("/")[-1]
    eval_model_without_org = args.eval_model.split("/")[-1]
    args.data_path = f"data/{args.scenario}/prompts.json"
    args.spec_path = f"data/{args.scenario}/specifications.json"
    args.response_save_path = f"result/{model_without_org}/{args.scenario}/{args.method}/generate.json"
    args.eval_save_path = f"result/{model_without_org}/{args.scenario}/{args.method}/{eval_model_without_org}_evaluate.json"
    args.score_save_path = f"result/{model_without_org}/{args.scenario}/{args.method}/{eval_model_without_org}_score.json"
    
    args.model_config_path = "config/config_model.yaml"
    args.ttd_config_path = "config/config_ttd.yaml"
    args.eval_config_path = "config/config_eval_model.yaml"
    
    print(f"Mode: {args.mode}")
    if args.mode == "generation":
        print(f"Data path: {args.data_path}")
    print(f"Specification path: {args.spec_path}")
    print(f"Response save path: {args.response_save_path}")
    if args.mode == "evaluation":
        print(f"Evaluation save path: {args.eval_save_path}")
        print(f"Score save path: {args.score_save_path}")
    
    return args

def wrap_inference(
    prompt,
    ttd_engine,
    **kwargs
):
    resp_dict = ttd_engine.inference(prompt)
    return resp_dict, kwargs

def run(args):
    if args.mode == "generation":
        llm_engine = get_llm_engine(
            model=args.model,
            provider=args.provider,
            ip=args.ip,
            port=args.port,
            config_path=args.model_config_path,
        )
        
        generate_response(
            data_path=os.path.expanduser(args.data_path), 
            spec_path=os.path.expanduser(args.spec_path), 
            save_path=os.path.expanduser(args.response_save_path),
            llm_engine=llm_engine,
            method=args.method,
            adversarial=args.adversarial,
            ttd_config_path=args.ttd_config_path,
            num_threads=args.num_threads,
        )
    
    if args.mode == "evaluation":
        eval_llm_engine = get_llm_engine(
            model=args.eval_model,
            provider=args.eval_model_provider,
            ip=args.ip,
            port=args.port,
            config_path=args.eval_config_path
        )
        
        evaluate_response(
            data_path=os.path.expanduser(args.response_save_path), 
            spec_path=args.spec_path,
            eval_save_path=os.path.expanduser(args.eval_save_path),
            score_save_path=os.path.expanduser(args.score_save_path),
            llm_engine=eval_llm_engine,
            evaluation_type=args.evaluation_type,
            num_threads=args.num_threads,
        )

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.ERROR,
        format="%(process)d-%(threadName)s-%(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()
    run(args)