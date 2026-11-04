# API Description

## Overview

Rec SDK Torch supports application development through Python interfaces. From a code invocation perspective, you can call all Python interfaces.

This section lists only the external interfaces that the service provides. Do not directly call other interfaces that are not described.

Rec SDK Torch is an extension based on the TorchRec APIs. Rec SDK Torch APIs depend on the classes and methods provided by TorchRec, but they do not support every TorchRec feature. This section describes the TorchRec API limitations that apply when you build models with Rec SDK Torch.

> [!NOTICE]
>
> 1. Because native TorchRec APIs are outside the management scope of Rec SDK Torch, parameter validity checks are not performed on the native TorchRec APIs used by the following APIs. You must ensure that the parameters are correct.
> 2. The current API list is organized by interface type. To view API usage by functional feature, see [Functions and Features](../04_migration_and_training/migration_and_training.md#functional_features_description).
> 3. Some API parameter descriptions contain the phrase "user-defined values are not supported," which means that the parameter does not support user-defined values and can only use the default value. If you pass a non-default value, an exception is thrown.
> 4. Some API parameter descriptions contain statements such as "supported only in TorchRec v1.2.0," which means that the parameter can be passed only when you use the corresponding TorchRec version.
> 5. For related terms, see [Core Terms](../01_introduction/introduction.md#core_terms).

## API Category Overview

The Rec SDK Torch APIs are categorized by interface type as follows. Interfaces marked with `(TorchRec)` in the documents are open-source TorchRec interfaces, and the rest are Rec SDK Torch interfaces:

| Category | Document | Description |
|------|------|------|
| Table Creation APIs | [table_creation_apis.md](02_table_creation_apis.md) | Configuration and creation of sparse tables, including two creation modes: full-device memory mode and multi-level cache mode |
| Data APIs | [data_apis.md](03_data_apis.md) | Representation and encapsulation of sparse data, used to pass query IDs and feature information to sparse tables |
| Sharding APIs | [subtable_apis.md](05_subtable_apis.md) | Sharding policies and execution for sparse tables in distributed environments |
| Optimizer APIs | [optimizers_apis.md](04_optimizers_apis.md) | Optimizer configuration for sparse table parameters, with support for gradient accumulation |
| Pipeline APIs | [pipeline_apis.md](06_pipeline_apis.md) | Creation and execution of training pipelines, supporting full-device memory and multi-level cache pipeline modes |
| Multi-Level Cache Management | [multilevel_cache_management_apis.md](07_multilevel_cache_management_apis.md) | Model save/load and weight initialization types in multi-level cache mode |
| Admission and Eviction Management | [access_and_elimination_management_apis.md](08_access_and_elimination_management_apis.md) | Configuration of admission and eviction policies for sparse table features |
| Custom Operators | [specialized_operator.md](09_specialized_operator.md) | Custom operators accelerated by NPU, some of which are bound to open-source APIs |

## Recommended Reading Order

If you are using Rec SDK Torch for the first time, you are advised to read the API documents in the following order:

1. **This page (API Description)** — Understand the overall API structure and core concepts.
2. **[Table Creation APIs](02_table_creation_apis.md)** — Understand how to configure and create sparse tables, the first step in building a model.
3. **[Data APIs](03_data_apis.md)** — Learn how to pass query data to sparse tables.
4. **[Sharding APIs](05_subtable_apis.md)** — Learn how to shard sparse tables in distributed environments.
5. **[Optimizer APIs](04_optimizers_apis.md)** — Configure optimizers for sparse table parameters.
6. **[Pipeline APIs](06_pipeline_apis.md)** — Create training pipelines, connecting table lookup with forward and backward propagation.
7. **[Multi-Level Cache Management](07_multilevel_cache_management_apis.md)**/[**Admission and Eviction Management](08_access_and_elimination_management_apis.md)** — Advanced features, read as needed.
8. **[Custom Operators](09_specialized_operator.md)** — Learn about the available NPU-accelerated operators.

## API Call Conventions

### Import Paths

The code examples in the documents use the following import path conventions:

- **Full-device memory mode interfaces**: Import them from `hybrid_torchrec`, for example, `from hybrid_torchrec import HashEmbeddingBagConfig`.
- **Multi-level cache mode interfaces**: Import them from `torchrec_embcache`, for example, `from torchrec_embcache.distributed import EmbCacheEmbeddingBagConfig`.
- **Native TorchRec interfaces**: Import them from `torchrec`, for example, `from torchrec.distributed.planner import EmbeddingShardingPlanner`.

### Return Value Conventions

- Success: Returns the corresponding object or `None`.
- Failure: An exception is thrown. For specific exception types and causes, see the parameter constraint descriptions of each API.
