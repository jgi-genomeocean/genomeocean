# ligase_h8
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


out=ligase-4b
model=DOEJGI/genomeocean-4b
python ../autocomplete_structure.py \
  --gen_id $gene --start $start --end $end --strand $strand \
  --prompt_start $pstart --prompt_end $pend --structure_start $sstart --structure_end $send \
  --num 30 --min_seq_len $min --max_seq_len $max \
  --foldmason_path /global/homes/z/zhwang/bin/foldmason --method foldmason \
  --output_prefix $out \
  --model_dir $model
