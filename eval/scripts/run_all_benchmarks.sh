#!/bin/bash
set -e

echo "> run_all_benchmarks.sh $@"

# Parse arguments
cuda_device="$1"
ckpt="$2"
conv_mode="$3"
question_extension="$4"

benchmarks=(
    gqa
    # vizwiz
    scienceqa
    textvqa
    # pope
    mme
    mmbench_en
    # mmbench_cn
    mmvet
    mmmu
    mathvista
    ai2d
    chartqa
    docvqa
    # infovqa
    # stvqa
    ocrbench
    mmstar
    realworldqa
    mmvp
    # synthdog
    seed
)

# Create a directory for checkpoint files if it doesn't exist
checkpoint_dir="checkpoints"
mkdir -p "$checkpoint_dir"

# Generate a unique checkpoint file name based on the script arguments
checkpoint_file="$checkpoint_dir/checkpoint_$(basename $ckpt)_$conv_mode.txt"
script_dir=$(dirname $(realpath $0))

# Check if the checkpoint file exists and load the completed benchmarks
if [[ -f "$checkpoint_file" ]]; then
    completed_benchmarks=($(cat "$checkpoint_file"))
    echo "Resuming from checkpoint. Completed benchmarks: ${completed_benchmarks[@]}"
else
    completed_benchmarks=()
fi

# Record start time in seconds since epoch
start_time=$(date +%s)
timestamp=$(date "+%Y-%m-%d %H:%M:%S")

for benchmark in "${benchmarks[@]}"; do
    if [[ " ${completed_benchmarks[@]} " =~ " $benchmark " ]]; then
        echo "Skipping completed benchmark: $benchmark"
        continue
    fi

    echo "Running benchmark: $benchmark"
    CUDA_VISIBLE_DEVICES=$cuda_device bash $script_dir/run_benchmark.sh \
        --benchmark $benchmark \
        --ckpt $ckpt \
        --conv_mode $conv_mode \
        --question_extension "$question_extension"
    echo "Finished benchmark: $benchmark"
    wait

    # Append the completed benchmark to the checkpoint file
    echo "$benchmark" >> "$checkpoint_file"

    # Calculate elapsed time properly
    current_time=$(date +%s)
    elapsed_minutes=$(( (current_time - start_time) / 60 ))
    echo "Elapsed minutes: $elapsed_minutes"
    echo ""
done

# Calculate final elapsed time
end_time=$(date +%s)
total_elapsed_minutes=$(( (end_time - start_time) / 60 ))
echo "Finished all benchmarks"
echo "Total elapsed minutes: $total_elapsed_minutes"