scenario=Child
model=Qwen3-14B-thinking
eval_model=GPT-4.1
eval_model_provider=openai
method=vanilla
num_threads=32

python src/run.py \
    --mode evaluation \
    --scenario $scenario \
    --model $model \
    --eval_model eval_model \
    --eval_model_provider eval_model_provider \
    --evaluation_type joint \
    --method $method \
    --num_threads 32 \
    --ip $ip \
    --port $port