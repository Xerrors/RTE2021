token=`date "+%Y-%m-%d_%H-%M-%S"`-temp
python -m train \
--ex_index=${token} \
--corpus_type=WebNLG \
--device_id=0 \
--epoch_num=150 \

wait