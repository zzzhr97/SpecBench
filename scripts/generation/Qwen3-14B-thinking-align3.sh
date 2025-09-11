scenario=Child
model=Qwen3-14B-thinking
provider=vllm-server
method=align3
num_threads=32

python src/run.py \
    --mode generation \
    --scenario $scenario \
    --model $model \
    --provider $provider \
    --method $method \
    --num_threads 32 \
    --ip $ip \
    --port $port