#!/bin/bash
#SBATCH --job-name=run_all
#SBATCH --output=run_all_%j.log
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --chdir=.
#SBATCH --cpus-per-task=4
#SBATCH --constraint=rtx_2080

# ==============================================================================
# Configuration
# ==============================================================================
shadow_model_path='./shadow_models'
target_model_path='./target_models'
epochs=10
removal=0    # Default removal value, if needed
num_shadow=4 # Number of shadow models to train
num_target=1 # Number of target models to train

# Set your user_id and rate explicitly here:
user_id=1
rate=0.05

# ==============================================================================
# Main Submission Logic
# ==============================================================================
echo "Starting jobs for user_id=${user_id} and rate=${rate}"

shadow_job_ids=""
target_job_ids=""

# ------------------------------------------------------------------------------
# 1) Submit SHADOW models
# ------------------------------------------------------------------------------
for (( i=0; i<${num_shadow}; i++ )); do
  shadow_model_seed=$i
  shadow_model_name="1m_Neu_MF_seed${i}"
  echo "Submitting shadow model $shadow_model_name (seed=$shadow_model_seed)"

  job_id=$(
    sbatch --parsable --job-name=shadow_${i}_user${user_id}_rate${rate} <<EOT
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --chdir=$(pwd)
#SBATCH --output=logs/shadow_${i}_user${user_id}_rate${rate}_%j.out
#SBATCH --error=logs/shadow_${i}_user${user_id}_rate${rate}_%j.err

module load anaconda3
source activate rec_sys

python train_user.py --epochs ${epochs} \
                     --model_name ${shadow_model_name} \
                     --model_path ${shadow_model_path} \
                     --seed ${shadow_model_seed}
EOT
  )
  # Accumulate shadow job IDs
  shadow_job_ids="${shadow_job_ids}:${job_id}"
done

# ------------------------------------------------------------------------------
# 2) Submit TARGET models
# ------------------------------------------------------------------------------
for (( i=0; i<${num_target}; i++ )); do
  target_model_seed=$i
  target_model_name="1m_Neu_MF_seed${i}"
  echo "Submitting target model $target_model_name (seed=$target_model_seed)"

  t_job_id=$(
    sbatch --parsable --gres=gpu:1 --mem=16G --time=02:00:00 \
           --chdir=$(pwd) --job-name=target_${i}_user${user_id}_rate${rate} <<EOT
#!/bin/bash
#SBATCH --output=logs/target_${i}_user${user_id}_rate${rate}_%j.out
#SBATCH --error=logs/target_${i}_user${user_id}_rate${rate}_%j.err

module load anaconda3
source activate rec_sys

python train_user.py --epochs ${epochs} \
                     --model_name ${target_model_name} \
                     --model_path ${target_model_path} \
                     --seed ${target_model_seed}
EOT
  )
  # Accumulate target job IDs
  target_job_ids="${target_job_ids}:${t_job_id}"
done

# ------------------------------------------------------------------------------
# 3) Post-processing job (generate_gaussian.py)
# ------------------------------------------------------------------------------
# Remove leading colons from job IDs
shadow_job_ids="${shadow_job_ids#:}"
target_job_ids="${target_job_ids#:}"

# Combine IDs for afterany dependency
all_job_ids="${shadow_job_ids}:${target_job_ids}"

echo "Submitting generate_gaussian.py job for user_id=${user_id}, rate=${rate}"

gaussian_job_id=$(
  sbatch --parsable \
         --dependency=afterany:${all_job_ids} \
         --gres=gpu:1 --mem=16G --time=00:20:00 \
         --chdir=$(pwd) \
         --job-name=generate_gaussian_user${user_id}_rate${rate} <<EOT
#!/bin/bash
#SBATCH --output=logs/generate_gaussian_user${user_id}_rate${rate}_%j.out
#SBATCH --error=logs/generate_gaussian_user${user_id}_rate${rate}_%j.err

module load anaconda3
source activate rec_sys

python generate_gaussian.py
EOT
)

echo "Shadow Job IDs:   ${shadow_job_ids}"
echo "Target Job IDs:   ${target_job_ids}"
echo "Gaussian Job ID:  ${gaussian_job_id}"
