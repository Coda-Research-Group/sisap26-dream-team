task=1
datasetName="wikipedia-small"
inputDataset="data/$datasetName"
outputDir="results/$datasetName"
mkdir -p $outputDir

echo Running Task $task
docker run \
    --rm \
    --user "$(id -u):$(id -g)" \
    --cpus=8 \
    --memory=24g \
    --memory-swap=24g \
    --memory-swappiness 0 \
    --volume $(pwd)/search.py:/app/search.py:ro \
    --volume $(pwd)/data:/app/data:ro \
    --volume $(pwd)/results:/app/results:rw \
    --entrypoint python3 \
    sisap-baseline \
    "/app/search.py" \
    --input $inputDataset/*.h5 \
    --task-description $inputDataset/config.json \
    --output $outputDir

task=2
datasetName="llama-dev"
inputDataset="data/$datasetName"
outputDir="results/$datasetName"
mkdir -p $outputDir

echo Running Task $task
docker run \
    --rm \
    --user "$(id -u):$(id -g)" \
    --cpus=8 \
    --memory=24g \
    --memory-swap=24g \
    --memory-swappiness 0 \
    --volume $(pwd)/search.py:/app/search.py:ro \
    --volume $(pwd)/data:/app/data:ro \
    --volume $(pwd)/results:/app/results:rw \
    --entrypoint python3 \
    sisap-baseline \
    "/app/search.py" \
    --input $inputDataset/*.h5 \
    --task-description $inputDataset/config.json \
    --output $outputDir
