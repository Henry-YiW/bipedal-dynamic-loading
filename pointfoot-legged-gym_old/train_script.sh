export ROBOT_TYPE=PF_TRON1A
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python legged_gym/scripts/train.py --task=pointfoot_rough_load --num_envs=4096 --max_iterations=7000 --headless