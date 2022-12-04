cd /p/reviewde/data/ratebeer/graph/small_500_v2/graph_batch/train

files=()
includes=()

for i in {0..1000}; do
    files+=( "$i.bin" )
done

cp -- "${files[@]}" ../train_debug/