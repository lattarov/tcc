source ~/miniconda3/etc/profile.d/conda.sh

conda activate tcc

python src/run_model.py RL

python src/run_model.py PID

python src/run_model.py RL -impulse

python src/run_model.py PID -impulse
