# variables
NUM_SERVERS=2
WORKERS_PER_SERVER=1
RANK=0
# network configurations
IP=127.0.0.1
PORT=8888
MASTER_ADDR=gpu015
MASTER_PORT=8888
MASTER_ADDR=127.0.0.1
MASTER_PORT=8888
# run the script
torchrun --nproc_per_node=$WORKERS_PER_SERVER --nnodes=$NUM_SERVERS --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py \
--dataset pubmed \
--num_parts $(($WORKERS_PER_SERVER*$NUM_SERVERS)) \
--backend gloo \
--init_method env:// \
--model_name sage \
--mode AdaQP \
--assign_scheme adaptive \
--logger_level INFO