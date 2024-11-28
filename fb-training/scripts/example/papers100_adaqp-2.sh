# variables
NUM_SERVERS=3
WORKERS_PER_SERVER=4
RANK=0
# network configurations
IP=10.100.20.15
PORT=29501
# run the script
torchrun --nproc_per_node=$WORKERS_PER_SERVER --nnodes=$NUM_SERVERS --node_rank=1 --master_addr=10.100.20.15 --master_port=$PORT main.py \
--dataset ogbn-papers100M \
--num_parts $(($WORKERS_PER_SERVER*$NUM_SERVERS)) \
--backend gloo \
--init_method tcp://10.100.20.15:29500 \
--model_name sage \
--mode AdaQP \
--assign_scheme adaptive \
--logger_level INFO
