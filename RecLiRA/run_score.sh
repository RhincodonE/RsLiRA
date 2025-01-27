#!/bin/bash
#SBATCH --job-name=run_all
#SBATCH --output=run_all_%j.log
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --chdir=.
#SBATCH --cpus-per-task=4
#SBATCH --constraint=rtx_2080

# Make sure logs directory exists
mkdir -p logs

# Define variables
shadow_model_path='./shadow_models'
target_model_path='./target_models'
epochs=20

# Number of shadow and target models
num_shadow=1
num_target=500

# Collect job IDs
shadow_job_ids=""
target_job_ids=""

# ------------------------------------------------------------------------------
# Submit Shadow Model Training Jobs
# ------------------------------------------------------------------------------
for (( i=0; i<$num_shadow; i++ )); do
  shadow_model_seed=$i
  shadow_model_name="NeuMF_shadow_seed${i}"

  echo "Submitting shadow model: $shadow_model_name (seed=$shadow_model_seed)"

  job_id=$(sbatch --parsable --job-name=shadow_${i} <<EOT
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --chdir=$(pwd)
#SBATCH --output=logs/shadow_${i}_%j.out
#SBATCH --error=logs/shadow_${i}_%j.err

module load anaconda3
source activate rec_sys

python train_user.py --epochs $epochs \
                     --model_name $shadow_model_name \
                     --model_path $shadow_model_path \
                     --seed $shadow_model_seed
EOT
)

  # Append this job ID (with a colon) to our list of shadow job IDs
  shadow_job_ids="$shadow_job_ids:$job_id"
done

# ------------------------------------------------------------------------------
# Submit Target Model Training Jobs
# ------------------------------------------------------------------------------
for (( i=0; i<$num_target; i++ )); do
  target_model_seed=$i
  target_model_name="NeuMF_target_seed${i}"

  echo "Submitting target model: $target_model_name (seed=$target_model_seed)"

  t_job_id=$(sbatch --parsable --gres=gpu:1 --mem=16G --time=02:00:00 \
      --chdir=$(pwd) --job-name=target_${i} <<EOT
#!/bin/bash
#SBATCH --output=logs/target_${i}_%j.out
#SBATCH --error=logs/target_${i}_%j.err

module load anaconda3
source activate rec_sys

python train_user.py --epochs $epochs \
                     --model_name $target_model_name \
                     --model_path $target_model_path \
                     --seed $target_model_seed
EOT
  )
  # Append target job ID
  target_job_ids="$target_job_ids:$t_job_id"
done

# ------------------------------------------------------------------------------
# Combine job IDs and remove leading colon
# ------------------------------------------------------------------------------
shadow_job_ids=${shadow_job_ids#:}
target_job_ids=${target_job_ids#:}
all_job_ids="$shadow_job_ids:$target_job_ids"

# ------------------------------------------------------------------------------
# Submit Job to Run generate_gaussian.py AFTER All Shadow & Target Jobs
# ------------------------------------------------------------------------------
echo "Submitting generate_gaussian.py job with dependency on all training jobs..."
gaussian_job_id=$(sbatch --parsable --dependency=afterany:$all_job_ids \
    --gres=gpu:1 --mem=16G --time=00:20:00 \
    --chdir=$(pwd) --job-name=generate_gaussian <<EOT
#!/bin/bash
#SBATCH --output=logs/generate_gaussian_%j.out
#SBATCH --error=logs/generate_gaussian_%j.err

module load anaconda3
source activate rec_sys

python generate_gaussian.py
EOT
)

# ------------------------------------------------------------------------------
# Submit score.py AFTER generate_gaussian.py
# ------------------------------------------------------------------------------
echo "Submitting score.py job with dependency on generate_gaussian.py..."
score_job_id=$(sbatch --parsable --dependency=afterany:$gaussian_job_id \
    --gres=gpu:1 --mem=16G --time=00:20:00 \
    --chdir=$(pwd) --job-name=score <<EOT
#!/bin/bash
#SBATCH --output=logs/score_%j.out
#SBATCH --error=logs/score_%j.err

module load anaconda3
source activate rec_sys

python score.py
EOT
)

echo "All jobs submitted!"
