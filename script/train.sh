token=`date "+%Y-%m-%d_%H-%M-%S"`-temp
python -m train \
--ex_index=${token} \
--corpus_type=WebNLG \
--device_id=1 \
--epoch_num=150 \
--use_feature_enchanced \

wait