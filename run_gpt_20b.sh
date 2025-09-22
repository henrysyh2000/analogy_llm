# load environment variables
source ~/env.sh

# ensure the subfolder exists
mkdir -p "$(./slurm_cli_log.py mkpath)"

# submit sbatch script
sbatch --output=$SLURM_PATH/%j_gpt_20b_analobench.out --error=$SLURM_PATH/%j_gpt_20b_analobench.err sbatch_script.slurm