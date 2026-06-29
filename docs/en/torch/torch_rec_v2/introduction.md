# Introduction

## Overview

Rec SDK Torch provides the following functions:

- Basic model training functions:
- Rec SDK supports single-node, single-card training and single-node, multi-card distributed training.
- Models developed based on Torch are supported.

- Recommendation-specific functions:

    Based on the dynamic sparse table solution, Rec SDK Torch provides essential recommendation functions, such as hash mapping, dynamic sparse table expansion, and dynamic sparse table operators implemented based on hierarchical key-value (HKV), a high-performance key-value storage acceleration library.

- Large-scale sparse table functions:

    Rec SDK Torch supports row-wise distributed sparse table sharding.

**Key features**

Rec SDK Torch provides capabilities such as hash mapping, row-wise table sharding, dynamic sparse table expansion and eviction, and dynamic sparse table operators.

- Hash mapping

    Torch provides `nn.Embedding` for dense ID lookup. In recommendation scenarios, most raw feature IDs are discrete, which makes direct lookup inconvenient. To address this issue, Rec SDK Torch provides hash mapping based on dynamic sparse tables. This function maps discrete feature keys to storage addresses in real time, so you do not need to convert IDs to contiguous values in advance.

- Row-wise sharding

    When embeddings are sharded across different tables, Rec SDK Torch partitions embeddings by row and uses a modulo-based bucketing strategy to determine the bucket position of an embedding in the table from the remainder of its ID.

- Dynamic expansion and eviction of sparse tables

    Sparse tables support elastic expansion. The system automatically allocates space when new features are added. When space usage reaches the threshold, the built-in eviction policy of the operator automatically clears low-frequency inactive features to ensure efficient use of storage resources.

- Dynamic sparse table operators

    Rec SDK Torch provides the deeply optimized, high-performance custom operator extension `dynamic_emb_extensions`. Compared with the native Torch implementation, it greatly improves training throughput.

## Software Architecture

**Figure 1** Software architecture
![](../../figures/torch_rec_v1/software-architecture.png "Software architecture")

Built upon TorchRec, mainstream recommendation frameworks, CANN, and diverse hardware and network architectures, Rec SDK Torch addresses the specific requirements of search, recommendation, and advertising model training. It provides high-performance, streamlined APIs designed for ease of use, enabling Ascend AI Processors to achieve highly efficient training for search, recommendation, and advertising models. Table 1 describes the modules.

**Table 1** Modules in the architecture diagram

|Rec SDK Torch Module|Description|
|--|--|
|Recommendation API layer|Provides easy-to-use APIs to simplify customer access and support service growth.|
|Recommended function layer|Provides core capabilities to meet customer requirements.|
|Recommendation acceleration layer|Provides core components to build performance competitiveness and offer superb performance for the entire system.|
|Recommendation storage layer|Supports distributed storage of sparse tables.|
|Torchrec-npu|Ascend adaptation layer for open-source TorchRec.|
