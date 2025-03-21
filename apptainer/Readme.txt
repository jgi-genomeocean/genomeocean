#To build
apptainer build --tmpdir $HOME/scratch/temp/ genomeocean.sif genomeocean.def
#To run
apptainer run --nv --bind $HOME/scratch/temp:/workspace genomeocean.sif
