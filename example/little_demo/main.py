# coding: UTF-8
import logging
import os
import warnings
from glob import glob
import tensorflow as tf

from config import sess_config, Config
from dataset import generate_dataset
from optimizer import get_dense_and_sparse_optimizer
from model import MyModel
from mx_rec.util.tf_version_adapter import hccl_ops
from mx_rec.core.asc.feature_spec import FeatureSpec
from mx_rec.core.asc.helper import get_asc_insert_func
from mx_rec.core.asc.manager import start_asc_pipeline
from mx_rec.core.embedding import create_table, sparse_lookup
from mx_rec.graph.modifier import modify_graph_and_start_emb_cache
from mx_rec.constants.constants import MxRecMode, ASCEND_TIMESTAMP
from mx_rec.util.initialize import get_rank_id, get_rank_size, init, clear_channel, terminate_config_initializer, \
    set_if_load, get_initializer
from mx_rec.util.variable import get_dense_and_sparse_variable

tf.compat.v1.disable_eager_execution()


def make_batch_and_iterator(is_training, feature_spec_list=None,
                            use_timestamp=False, dump_graph=False, batch_number=100):
    dataset = generate_dataset(cfg, use_timestamp=use_timestamp, batch_number=batch_number)
    if not MODIFY_GRAPH_FLAG:
        insert_fn = get_asc_insert_func(tgt_key_specs=feature_spec_list, is_training=is_training, dump_graph=dump_graph)
        dataset = dataset.map(insert_fn)
    dataset = dataset.prefetch(100)
    iterator = dataset.make_initializable_iterator()
    batch = iterator.get_next()
    return batch, iterator


def model_forward(input_list, batch, is_train, modify_graph, config_dict=None):
    embedding_list = []
    feature_list, hash_table_list, send_count_list = input_list
    for feature, hash_table, send_count in zip(feature_list, hash_table_list, send_count_list):
        access_and_evict_config = None
        if isinstance(config_dict, dict):
            access_and_evict_config = config_dict.get(hash_table.table_name)
        embedding = sparse_lookup(hash_table, feature, send_count, dim=None, is_train=is_train,
                                  access_and_evict_config=access_and_evict_config,
                                  name=hash_table.table_name + "_lookup", modify_graph=modify_graph, batch=batch)

        reduced_embedding = tf.reduce_sum(embedding, axis=1, keepdims=False)
        embedding_list.append(reduced_embedding)

    my_model = MyModel()
    my_model(embedding_list, batch["label_0"], batch["label_1"])
    return my_model


def build_graph(hash_table_list, is_train, feature_spec_list=None, config_dict=None, batch_number=100):
    batch, iterator = make_batch_and_iterator(is_train, feature_spec_list=feature_spec_list,
                                              use_timestamp=USE_TIMESTAMP, dump_graph=is_train,
                                              batch_number=batch_number)
    if MODIFY_GRAPH_FLAG:
        input_list = [[batch["user_ids"], batch["item_ids"]],
                      [hash_table_list[0], hash_table_list[1]],
                      [cfg.user_send_cnt, cfg.item_send_cnt]]
        if use_multi_lookup:
            input_list = [[batch["user_ids"], batch["item_ids"], batch["user_ids"], batch["item_ids"]],
                          [hash_table_list[0], hash_table_list[0], hash_table_list[0], hash_table_list[1]],
                          [cfg.user_send_cnt, cfg.item_send_cnt, cfg.user_send_cnt, cfg.item_send_cnt]]
        if USE_TIMESTAMP:
            tf.compat.v1.add_to_collection(ASCEND_TIMESTAMP, batch["timestamp"])
        model = model_forward(input_list, batch,
                              is_train=is_train, modify_graph=True, config_dict=config_dict)
    else:
        input_list = [feature_spec_list,
                      [hash_table_list[0], hash_table_list[1]],
                      [cfg.user_send_cnt, cfg.item_send_cnt]]
        if use_multi_lookup:
            input_list = [feature_spec_list,
                          [hash_table_list[0], hash_table_list[1], hash_table_list[0], hash_table_list[0]],
                          [cfg.user_send_cnt, cfg.item_send_cnt, cfg.user_send_cnt, cfg.item_send_cnt]]
        model = model_forward(input_list, batch,
                              is_train=is_train, modify_graph=False, config_dict=config_dict)

    return iterator, model


def evaluate():
    if MODIFY_GRAPH_FLAG:
        sess.run(get_initializer(False))
    else:
        sess.run(eval_iterator.initializer)
    clear_channel(is_train_channel=False)
    for j in range(1, EVAL_STEPS + 1):
        logging.info(f"################    eval at step {j} epoch {EPOCH}    ################")
        try:
            sess.run(eval_model.loss_list)
        except tf.errors.OutOfRangeError:
            logging.info(f"Encounter the end of Sequence for eval.")
            break


def create_feature_spec_list(use_timestamp=False):
    access_threshold = cfg.access_threshold if use_timestamp else None
    eviction_threshold = cfg.eviction_threshold if use_timestamp else None
    feature_spec_list = [FeatureSpec("user_ids", feat_count=cfg.user_feat_cnt, table_name="user_table",
                                     access_threshold=access_threshold,
                                     eviction_threshold=eviction_threshold),
                         FeatureSpec("item_ids", feat_count=cfg.item_feat_cnt, table_name="item_table",
                                     access_threshold=access_threshold,
                                     eviction_threshold=eviction_threshold)]
    if use_multi_lookup:
        feature_spec_list.extend([FeatureSpec("user_ids", feat_count=cfg.user_feat_cnt, table_name="user_table",
                                              access_threshold=access_threshold,
                                              eviction_threshold=eviction_threshold),
                                  FeatureSpec("item_ids", feat_count=cfg.item_feat_cnt, table_name="user_table",
                                              access_threshold=access_threshold,
                                              eviction_threshold=eviction_threshold)])
    if use_timestamp:
        feature_spec_list.append(FeatureSpec("timestamp", is_timestamp=True))
    return feature_spec_list


if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    warnings.filterwarnings("ignore")

    mode = MxRecMode.mapping(os.getenv("MXREC_MODE"))
    TRAIN_INTERVAL = 100
    EVAL_STEPS = 10
    SAVING_INTERVAL = 100

    # get init configuration
    use_mpi = bool(int(os.getenv("USE_MPI", 1)))
    use_dynamic = int(os.getenv("USE_DYNAMIC", 0))
    use_hot = bool(int(os.getenv("USE_HOT", 0)))
    use_dynamic_expansion = bool(int(os.getenv("USE_DYNAMIC_EXPANSION", 0)))
    use_multi_lookup = bool(int(os.getenv("USE_MULTI_LOOKUP", 1)))
    MODIFY_GRAPH_FLAG = bool(int(os.getenv("USE_MODIFY_GRAPH", 0)))
    USE_TIMESTAMP = bool(int(os.getenv("USE_TIMESTAMP", 0)))

    # nbatch function needs to be used together with the prefetch and host_vocabulary_size != 0
    init(use_mpi=use_mpi,
         train_interval=TRAIN_INTERVAL,
         eval_steps=EVAL_STEPS,
         prefetch_batch_number=5,
         use_dynamic=use_dynamic,
         use_hot=use_hot,
         use_dynamic_expansion=use_dynamic_expansion)
    IF_LOAD = False
    rank_id = get_rank_id()
    filelist = glob(f"./saved-model/sparse-model-{rank_id}-0")
    if filelist:
        IF_LOAD = True
    set_if_load(IF_LOAD)

    cfg = Config()
    # access_threshold unit counts; eviction_threshold unit seconds
    ACCESS_AND_EVICT = None
    if USE_TIMESTAMP:
        config_for_user_table = dict(access_threshold=cfg.access_threshold, eviction_threshold=cfg.eviction_threshold)
        config_for_item_table = dict(access_threshold=cfg.access_threshold, eviction_threshold=cfg.eviction_threshold)
        ACCESS_AND_EVICT = dict(user_table=config_for_user_table, item_table=config_for_item_table)
    train_feature_spec_list = create_feature_spec_list(use_timestamp=USE_TIMESTAMP)
    eval_feature_spec_list = create_feature_spec_list(use_timestamp=USE_TIMESTAMP)

    optimizer_list = [get_dense_and_sparse_optimizer(cfg) for _ in range(2)]
    sparse_optimizer_list = [sparse_optimizer for dense_optimizer, sparse_optimizer in optimizer_list]

    user_hashtable = create_table(key_dtype=tf.int64,
                                  dim=tf.TensorShape([cfg.user_hashtable_dim]),
                                  name='user_table',
                                  emb_initializer=tf.compat.v1.truncated_normal_initializer(),
                                  device_vocabulary_size=cfg.user_vocab_size * 10,
                                  host_vocabulary_size=0,  # cfg.user_vocab_size * 100, # for h2d test
                                  optimizer_list=sparse_optimizer_list,
                                  mode=mode)

    item_hashtable = create_table(key_dtype=tf.int64,
                                  dim=tf.TensorShape([cfg.item_hashtable_dim]),
                                  name='item_table',
                                  emb_initializer=tf.compat.v1.truncated_normal_initializer(),
                                  device_vocabulary_size=cfg.item_vocab_size * 10,
                                  host_vocabulary_size=0,  # cfg.user_vocab_size * 100, # for h2d test
                                  optimizer_list=sparse_optimizer_list,
                                  mode=mode)

    train_iterator, train_model = build_graph([user_hashtable, item_hashtable], is_train=True,
                                              feature_spec_list=train_feature_spec_list,
                                              config_dict=ACCESS_AND_EVICT, batch_number=cfg.batch_number)
    eval_iterator, eval_model = build_graph([user_hashtable, item_hashtable], is_train=False,
                                            feature_spec_list=eval_feature_spec_list,
                                            config_dict=ACCESS_AND_EVICT, batch_number=cfg.batch_number)
    dense_variables, sparse_variables = get_dense_and_sparse_variable()

    rank_size = get_rank_size()
    train_ops = []
    # multi task training
    for loss, (dense_optimizer, sparse_optimizer) in zip(train_model.loss_list, optimizer_list):
        # do dense optimization
        grads = dense_optimizer.compute_gradients(loss, var_list=dense_variables)
        avg_grads = []
        for grad, var in grads:
            if rank_size > 1:
                grad = hccl_ops.allreduce(grad, "sum") if grad is not None else None
            if grad is not None:
                avg_grads.append((grad, var))
        # apply gradients: update variables
        train_ops.append(dense_optimizer.apply_gradients(avg_grads))

        if use_dynamic_expansion:
            from mx_rec.constants.constants import ASCEND_SPARSE_LOOKUP_LOCAL_EMB, ASCEND_SPARSE_LOOKUP_ID_OFFSET

            train_emb_list = tf.compat.v1.get_collection(ASCEND_SPARSE_LOOKUP_LOCAL_EMB)
            train_address_list = tf.compat.v1.get_collection(ASCEND_SPARSE_LOOKUP_ID_OFFSET)
            # do sparse optimization by addr
            local_grads = tf.gradients(loss, train_emb_list)  # local_embedding
            grads_and_vars = [(grad, address) for grad, address in zip(local_grads, train_address_list)]
            train_ops.append(sparse_optimizer.apply_gradients(grads_and_vars))
        else:
            # do sparse optimization
            sparse_grads = tf.gradients(loss, sparse_variables)
            grads_and_vars = [(grad, variable) for grad, variable in zip(sparse_grads, sparse_variables)]
            train_ops.append(sparse_optimizer.apply_gradients(grads_and_vars))

    saver = tf.compat.v1.train.Saver()
    if MODIFY_GRAPH_FLAG:
        logging.info("start to modifying graph")
        modify_graph_and_start_emb_cache(dump_graph=True)
    else:
        start_asc_pipeline()

    with tf.compat.v1.Session(config=sess_config(dump_data=False)) as sess:
        if MODIFY_GRAPH_FLAG:
            sess.run(get_initializer(True))
        else:
            sess.run(train_iterator.initializer)
        sess.run(tf.compat.v1.global_variables_initializer())
        EPOCH = 0
        if os.path.exists(f"./saved-model/sparse-model-{rank_id}-%d" % 0):
            saver.restore(sess, f"./saved-model/model-{rank_id}-%d" % 0)
        else:
            saver.save(sess, f"./saved-model/model-{rank_id}", global_step=0)

        for i in range(1, 201):
            logging.info(f"################    training at step {i}    ################")
            try:
                sess.run([train_ops, train_model.loss_list])
            except tf.errors.OutOfRangeError:
                logging.info(f"Encounter the end of Sequence for training.")
                break
            else:
                if i % TRAIN_INTERVAL == 0:
                    EPOCH += 1
                    evaluate()

                if i % SAVING_INTERVAL == 0:
                    saver.save(sess, f"./saved-model/model-{rank_id}", global_step=i)

        saver.save(sess, f"./saved-model/model-{rank_id}", global_step=i)

    terminate_config_initializer()
    logging.info("Demo done!")
