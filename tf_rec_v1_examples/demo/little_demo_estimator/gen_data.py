from functools import lru_cache

import numpy as np

from config import GLOBAL_RANDOM_SEED


@lru_cache(maxsize=None)
def data_generate():
    """
    please copy parameter from
    1. run.sh
    2. main.py
    3. config.py
    """
    ret = []
    
    # from main.py
    max_data_generate_steps = 200
    
    # from run.sh
    rank_size = 8
    
    # from config.py
    batch_size = 4096
    item_range = 80000 * rank_size
    user_range = 200000 * rank_size
    category_range = 5000 * rank_size
    item_feat_cnt = 16
    user_feat_cnt = 8
    category_feat_cnt = 3
    
    # start generate
    batch_number = max_data_generate_steps * rank_size
    i = 0
    rs = np.random.RandomState(GLOBAL_RANDOM_SEED)
    while i < batch_number:
        item_ids = rs.randint(0, item_range, (batch_size, item_feat_cnt))
        user_ids = rs.randint(0, user_range, (batch_size, user_feat_cnt))
        category_ids = rs.randint(0, category_range, (batch_size, category_feat_cnt))
        label_0 = rs.randint(0, 2, (batch_size,))
        label_1 = rs.randint(0, 2, (batch_size,))
        
        ret.append({
            "item_ids": item_ids,
            "user_ids": user_ids,
            "category_ids": category_ids,
            "label_0": label_0,
            "label_1": label_1
        })
        i += 1
    return ret
