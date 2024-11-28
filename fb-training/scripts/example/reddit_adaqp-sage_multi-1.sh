# variables
NUM_SERVERS=2
WORKERS_PER_SERVER=1
RANK=0
# network configurations
IP=10.100.20.15
PORT=29501
# run the script
torchrun --nproc_per_node=$WORKERS_PER_SERVER --nnodes=$NUM_SERVERS --node_rank=0 --master_addr=10.100.20.15 --master_port=$PORT main.py \
--dataset reddit \
--num_parts $(($WORKERS_PER_SERVER*$NUM_SERVERS)) \
--backend gloo \
--init_method tcp://10.100.20.15:29500 \
--model_name gcn \
--mode AdaQP \
--assign_scheme adaptive \
--logger_level INFO