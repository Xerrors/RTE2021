# load args
# parser = argparse.ArgumentParser()
# parser.add_argument('--seed', type=int, default=2021, help="random seed for initialization")
# parser.add_argument('--ex_index', type=str, default="deafult")
# parser.add_argument('--corpus_type', type=str, default="WebNLG", help="NYT, WebNLG, NYT*, WebNLG*")
# parser.add_argument('--device_id', type=int, default=0, help="GPU index")
# parser.add_argument('--epoch_num', type=int, default=10, help="number of epochs")
# parser.add_argument('--multi_gpu', action='store_true', help="ensure multi-gpu training")
# parser.add_argument('--restore_file', default=None, help="name of the file containing weights to reload")

# parser.add_argument('--corres_threshold', type=float, default=0.5, help="threshold of global correspondence")
# parser.add_argument('--rel_threshold', type=float, default=0.5, help="threshold of relation judgement")
# parser.add_argument('--emb_fusion', type=str, default="concat", help="way to embedding")

# parser.add_argument('--ensure_cross', type=str, default='none', help="['none'(default),, 'avg']")
# parser.add_argument('--ensure_rel', type=str, default='default', help="['default', 'hgat', '_astoken', '_relemb']")
# parser.add_argument('--ensure_corres', type=str, default='default', help="correspondence ablation")
# parser.add_argument('--use_symmetries', type=str, default='none', help="symmetries relations, ['none'(default), 'symmetries', 'symmetries_rate']")
# parser.add_argument('--sent_rels', type=str, default='none', help="sentense relations, ['none'(default), 'global', 'contextual', 'contextual_global']")
# parser.add_argument('--sent_attn', type=str, default='none', help="sentense attention, ['none'(default)]")
# parser.add_argument('--cross_data', type=int, default=0, help="Extract implied relation")

token=`date "+%Y-%m-%d_%H-%M-%S"`
python -m train \
--ex_index=${token} \
--corpus_type=WebNLG \
--device_id=1 \
--epoch_num=100 \
--cross_data=1 \

# token=`date "+%Y-%m-%d_%H-%M-%S"`
# python -m train \
# --ex_index=${token} \
# --corpus_type=WebNLG \
# --device_id=0 \
# --epoch_num=100 \
# --cross_data=2 \


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
--sent_rels=global \
--sent_attn=none \

