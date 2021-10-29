#!/bin/bash
pypath=$(dirname $(dirname $(readlink -f $0)))
python $pypath/evaluate.py \
--ex_index=2021-10-01_21-46-53 \
--device_id=3 \
--mode=test \
--corpus_type=WebNLG-star \
--ensure_corres=default \
--ensure_rel=default \
--ensure_cross=none

wait

#!/bin/bash
pypath=$(dirname $(dirname $(readlink -f $0)))
python $pypath/evaluate.py \
--ex_index=2021-10-02_09-09-24 \
--device_id=3 \
--mode=test \
--corpus_type=WebNLG-star \
--ensure_corres=default \
--ensure_rel=hgat \
--ensure_cross=none

wait

#!/bin/bash
pypath=$(dirname $(dirname $(readlink -f $0)))
python $pypath/evaluate.py \
--ex_index=2021-10-02_22-59-32 \
--device_id=3 \
--mode=test \
--corpus_type=WebNLG-star \
--ensure_corres=default \
--ensure_rel=hgat_astoken \
--ensure_cross=none

wait


#!/bin/bash
pypath=$(dirname $(dirname $(readlink -f $0)))
python $pypath/evaluate.py \
--ex_index=2021-10-03_09-08-42 \
--device_id=3 \
--mode=test \
--corpus_type=WebNLG-star \
--ensure_corres=default \
--ensure_rel=hgat \
--ensure_cross=none

wait

#!/bin/bash
pypath=$(dirname $(dirname $(readlink -f $0)))
python $pypath/evaluate.py \
--ex_index=2021-10-03_22-03-36 \
--device_id=3 \
--mode=test \
--corpus_type=WebNLG-star \
--ensure_corres=default \
--ensure_rel=hgat_relemb_astoken \
--ensure_cross=none

wait

#!/bin/bash
pypath=$(dirname $(dirname $(readlink -f $0)))
python $pypath/evaluate.py \
--ex_index=2021-10-04_11-07-50 \
--device_id=3 \
--mode=test \
--corpus_type=WebNLG-star \
--ensure_corres=default \
--ensure_rel=hgat \
--ensure_cross=avg
