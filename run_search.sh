# TODO: Update hardware limits

task=1
dataset="wikipedia-small"
mkdir -p results/$dataset

echo Running Task $task
docker run \
    --rm \
    --user "$(id -u):$(id -g)" \
    --cpus=4 \
    --memory=16g \
    --memory-swap=16g \
    --memory-swappiness 0 \
    --volume $(pwd)/search.py:/app/search.py:ro \
    --volume $(pwd)/data:/app/data:ro \
    --volume $(pwd)/results:/app/results:rw \
    sisap-baseline python search.py --input data/$dataset/*.h5 --task-description data/$dataset/config.json --output results/$dataset/


task=2
dataset="llama-dev"
mkdir -p results/$dataset

echo Running Task $task
docker run \
    --rm \
    --user "$(id -u):$(id -g)" \
    --cpus=4 \
    --memory=16g \
    --memory-swap=16g \
    --memory-swappiness 0 \
    --volume $(pwd)/search.py:/app/search.py:ro \
    --volume $(pwd)/data:/app/data:ro \
    --volume $(pwd)/results:/app/results:rw \
    sisap-baseline python search.py --input data/$dataset/*.h5 --task-description data/$dataset/config.json --output results/$dataset/
