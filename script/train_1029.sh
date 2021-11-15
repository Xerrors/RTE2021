token=`date "+%Y-%m-%d_%H-%M-%S"`
python -m train \
--ex_index=${token} \
--corpus_type=WebNLG \
--device_id=1 \
--epoch_num=150 \
--ensure_cross=none \
--ensure_rel=default \
--ensure_corres=default \
--use_symmetries=none \
--sent_rels=global2 \
--sent_attn=none \

wait

token=`date "+%Y-%m-%d_%H-%M-%S"`
python -m train \
--ex_index=${token} \
--corpus_type=WebNLG \
--device_id=1 \
--epoch_num=150 \
--ensure_cross=none \
--ensure_rel=default \
--ensure_corres=default \
--use_symmetries=symmetries \
--sent_rels=global2 \
--sent_attn=none \