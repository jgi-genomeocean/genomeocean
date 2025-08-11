# lignase_h8
gene=NW_009276944.1
start=782277
end=783406
strand=-1
pstart=0
pend=450
sstart=0
send=352
min=250
max=300
foldmason=/global/homes/z/zhwang/bin/foldmason


out=$3
model=$1
num=$2
python ../1_generate_for_structure.py \
  --gen_id $gene --start $start --end $end --strand $strand \
  --prompt_start $pstart --prompt_end $pend \
  --num $num --min_seq_len $min --max_seq_len $max \
  --output_prefix $out \
  --model_dir $model

python ../lddt_scoring.py \
    --generated_seqs_csv $out.csv \
    --gen_id $gene \
    --start $start \
    --end $end \
    --strand $strand \
    --structure_start $pstart \
    --structure_end $pend \
    --foldmason_path $foldmason \
    --output_prefix $out 