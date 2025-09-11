scenario=Child
model=Qwen3-14B
provider=vllm-server
method=vanilla
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