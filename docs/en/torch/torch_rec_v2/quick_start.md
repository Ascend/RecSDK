# Quick Start

## Before You Start

This section provides a small demo that guides you through building a model with Rec SDK Torch based on a dynamic sparse table. The demo is available at [Rec SDK Torch Little Demo Sample](https://gitcode.com/Ascend/RecSDK/tree/develop_examples_and_tools/torch_rec_v2_examples/little_demo).

**Table 1**  little-demo file description

|File|Description|
|--|---|
|main.py|Entry point for model training|
|dataset.py|Dataset parsing and KeyedJaggedTensor data construction|
|model.py|Model file|
|run.sh|Startup script|
|logger.py|Log module definition|
|README.md|Dataset download and demo model running instructions|

## Interface Call Introduction

The following steps omit implementation details. For the complete code, see [Rec SDK Torch Little Demo Sample](https://gitcode.com/Ascend/RecSDK/tree/develop_examples_and_tools/torch_rec_v2_examples/little_demo). The key steps are as follows:

1. <a id="li524311501994"></a>Define the dataset and data conversion.

    A custom `Dataset` reads the raw data and uses `collate_fn` to convert sparse features into the `KeyedJaggedTensor` (KJT) format required by TorchRec.

    ```python
    # dataset.py
    def collate_fn(batch):
        ...
        kjt = KeyedJaggedTensor(
            keys=list(values.keys()),
            values=torch.cat(list(values.values())),
            lengths=torch.cat(list(lengths.values())),
        )
        return kjt, torch.tensor(labels, dtype=torch.float)

    # main.py
    train_loader = DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        ...
    )
    ```

2. Initialize the distributed environment.

    ```python
    # main.py
    backend = "hccl"
    dist.init_process_group(backend=backend)
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    ```

3. Defining the model structure.

    Build a model that includes an `EmbeddingCollection`.

    ```python
    # model.py
    class MovieLensModel(nn.Module):
        def __init__(self, embedding_module, ...):
            super().__init__()
            ...
            self.embedding_module = embedding_module
            self.over_arch = nn.Sequential(...)

        def forward(self, kjt: KeyedJaggedTensor) -> torch.Tensor:
            ...
            embeddings = self.embedding_module(kjt)
            ...
            return prediction
    ```

4. Configure `DynamicEmbeddingCollectionSharder`.
    Create a `DynamicEmbeddingCollectionSharder` and configure the fused optimizer parameters.

    ```python
    # model.py
    def get_sharder(args, optimizer_type):
        ...
        optimizer_kwargs = {
            "optimizer": optimizer_type,
            "learning_rate": learning_rate,
            ...
        }
        ...
        return DynamicEmbeddingCollectionSharder(
            fused_params=fused_params,
            use_index_dedup=True,
        )
    ```

5. Generate an automatic table sharding plan (planner).

    Use `DynamicEmbeddingShardingPlanner` with the network topology and table configuration to generate a distributed sharding plan.

    ```python
    # model.py
    def get_planner(device, eb_configs, batch_size, training):
        ···
        return DynamicEmbeddingShardingPlanner(
            eb_configs=eb_configs,
            topology=topology,
            constraints=dict_const,
            batch_size=batch_size,
            enumerator=enumerator,
        )
    ```

6. Build the distributed model.

    Wrap the original model with `DistributedModelParallel` (DMP).

    ```python
    # model.py
    def apply_dmp(model, args, training, device):
        ···
        plan = planner.collective_plan(model, [sharder], dist.GroupMember.WORLD)
        return DistributedModelParallel(
            module=model,
            device=device,
            sharders=[sharder],
            plan=plan,
        )
    ```1

7. Save and load the model state.

    Save the dense model weights and dynamic embedding data.
    Use dedicated APIs for sparse weights and optimizer state.
    - Saving: `DynamicEmbDump(save_dir, model, optim=True)`
    - Loading: `DynamicEmbLoad(save_dir, model, optim=True)`

## Model Training Startup

Model training depends on a container environment. You can quickly start the container and train the model using a prebuilt container image. You can also manually install dependencies and deploy the software in the container image environment.

### Option 1: Starting Model Training Using an Existing Container Image

1. Prepare Data

    Download the MovieLens-1M dataset and extract it to the ml-1m folder in the current directory.

2. Run run.sh directly

    The `run.sh` script automatically sets `PYTHONPATH` and starts training.

    ```bash
    bash run.sh
    ```
