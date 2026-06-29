# Introduction

## Overview

Rec SDK TensorFlow provides the following functions:

- Basic model training functions: Rec SDK TensorFlow supports single-server single-card training, single-server multi-card training, and multi-server multi-card distributed training, as well as model development based on TensorFlow.
- Recommendation-specific functions: Based on the sparse table solution, Rec SDK TensorFlow provides essential functions such as feature saving and loading, feature admission, feature eviction, and non-affinity operator splitting.
- Large-scale sparse table functions: Rec SDK TensorFlow supports multi-level storage across accelerator card memory, host memory, and host drive. It also supports storage and dynamic scaling, with capacities exceeding 10 TB.
- Customized model training functions: Rec SDK TensorFlow supports customized WarmStart options and loads parameters from multiple source-domain models to achieve continuous transfer learning.
- Performance and accuracy detection tools: Performance detection tools support host-side profiling data collection, fusion of host-side and device-side profiling data, time consumption sorting, and visualization, facilitating performance issue localization. Accuracy monitoring tools support end-to-end operator-level accuracy comparison, facilitating accuracy maintenance and issue localization.

**Key features**

Rec SDK TensorFlow provides functional features such as dynamic expansion, dynamic shapes, automatic graph modification, feature admission and eviction, Hot_Embedding, and customized WarmStart. You can incorporate the desired features into your adapted models.

- Dynamic expansion

    TensorFlow accommodates embeddings through variables. You need to estimate the size of each table and then create variables through APIs. The size of the embedding table, once set, cannot be increased or reduced later. This may cause either a waste of NPU memory or insufficient space. In recommendation scenarios, the size of multiple sparse tables is often unpredictable. To better meet user requirements, a dynamic capacity expansion feature for sparse tables has been added. On-chip memory supports both dynamic and static capacity expansion; with dynamic on-chip memory scaling, device memory usage grows during training. DDR/SSD modes only support dynamic scaling, where host memory/drive usage grows while device memory remains constant.

    For details about the process, see [Dynamic Expansion Mode on On-chip Memory](appendix.md#on-chip-memory-dynamic-expansion-mode).

    Sparse table operator samples and README files can be obtained from this [link](https://gitcode.com/Ascend/RecSDK/tree/develop/cust_op/ascendc_op/ai_core_op/cust_op_by_addr/v220).

    >[!NOTE]
    >When dynamic capacity expansion is enabled, use compatible optimizers such as SGDByAddr, LazyAdamByAddress, and AdagradByAddress.

- Dynamic shapes

    The Rec SDK TensorFlow training framework supports dynamic shapes. The shapes in TensorFlow depend on specific operations. Both operator inputs and outputs are dynamic shapes.

    For details about the process, see [Dynamic Shapes](appendix.md#dynamic-shapes).

- Automatic graph modification

    Rec SDK TensorFlow trains features either by creating the FeatureSpec class or automatically modifying the TensorFlow computational graph.

    Automatic graph modification modifies the TensorFlow graph so that training scripts do not require FeatureSpec creation or explicit call functions that embed read embedding key operators.

    For details about the process, see [Automatic Graph Modification](appendix.md#automatic-graph-modification).

    >[!NOTE]
    >Currently, automatic graph modification can be performed only in the default graph of TensorFlow. You cannot create a `tf.Graph` class to define the computational graph for your model.

- Feature admission and eviction

    Low-frequency features often do not help training, causing memory waste and overfitting. To address this problem, the feature admission function is designed to filter out the features with low frequency. Feature access and eviction can be used in FeatureSpec mode or automatic graph modification mode.

    Features that are not helpful to training need to be eliminated to prevent them from affecting the training effect and to save memory. Rec SDK TensorFlow supports two eviction triggers: global step intervals and time intervals.

    For details about the process, see [Feature Admission and Eviction](appendix.md#feature-admission-and-eviction).

    >[!NOTE]
    >Currently, you can enable admission alone or both admission and eviction together. Enabling eviction alone is not supported.

- Hot\_Embedding

    In recommendation scenarios with high key repetition rates, Hot_Embedding caches frequently access keys to accelerate table lookups.

- Customized WarmStart

    In TensorFlow Estimator mode, native WarmStart supports loading partial or full model parameters from a single path when training a new model. This function provides a more flexible way to restore model parameters. WarmStart is commonly used in transfer learning. That is, when a model trained on a task is used for another task, some layers or parameters of the model can be reused to accelerate the learning process of the new task. Name mapping for embedding tables is currently not supported.

    Customized WarmStart features:

    - Compatibility with native TF WarmStart functionality and WarmStart support for specified sparse tables.
    - Multi-path WarmStart support and loading partial or full model parameters from multiple model paths to support the training of multi-model transfer learning tasks

    >[!NOTE]
    >Customized WarmStart is only supported in on-chip memory and DDR modes under TensorFlow 1.15.0.

- Incremental model saving and loading

    This supports streaming training. Recommendation systems continuously generate log data for Click-Through Rate (CTR) models. The model is trained with the received data, and the full or incremental model is saved at a certain interval.

    Saving only incremental updates for sparse parameters significantly reduces the overhead of frequent model checkpoints. This allows a model to be restored using the latest full checkpoint combined with a series of incremental checkpoints, reducing redundant computation.

    >[!NOTE]
    >- Incremental model saving and loading support only the on-chip memory mode, DDR mode, SSD mode, and Estimator mode in training and prediction modes, and do not support `train_and_evaluate`. Only capacity expansion and non-capacity expansion scenarios are supported.
    >- Incremental model saving and loading cannot be enabled together with feature admission and eviction.

- Multi-lookup for single tables

    It consolidates multiple queries to a single sparse table into a single lookup operation.

- PCIe through

    It utilizes PCIe-through pipelined parallel swapping (in/out) and shared memory for data exchange, increasing throughput between the host and device.

## Software Architecture

![](../../figures/tf_rec_v1/4-mxRec-architecture.png)

Built upon mainstream recommendation frameworks, CANN, and diverse hardware and network architectures, Rec SDK TensorFlow addresses the specific requirements of search, recommendation, and advertising model training. It provides high-performance, streamlined APIs designed for ease of use, enabling Ascend AI Processors to achieve highly efficient training for search, recommendation, and advertising models.

**Table 1**  Modules in the architecture diagram

|Rec SDK TensorFlow Module|Description|
|--|--|
|API layer|Provides easy-to-use APIs to simplify customer access and support service growth.|
|Recommended function layer|Provides core capabilities to meet customer requirements.|
|Recommendation acceleration layer|Provides core components to build performance competitiveness and offer superb performance for the entire system.|
|Sparse storage layer|Supports large-scale sparse table storage of more than 10 TB.|

## Supported Hardware and OSs

**Table 2** Supported products

|Product|Architecture|OS Version|
|--|--|--|
|Atlas 800T A2 training server<br>Atlas 200T A2 Box16 heterogeneous subrack|<li>Arm</li><li>x86_64</li>|<li>CentOS 7.6</li><li>openEuler 22.03</li><li>Ubuntu 20.04</li>|
|Atlas 900 A3 SuperPoD|Arm|openEuler 22.03|
