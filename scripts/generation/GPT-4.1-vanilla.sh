scenario=Child
model=GPT-4.1
provider=openai
method=vanilla
num_threads=32

python src/run.py \
    --mode generation \
    --scenario $scenario \
    --model $model \
    --provider $provider \
    --method $method \
    --num_threads 32