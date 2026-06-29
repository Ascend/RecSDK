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

1. Define the dataset and data conversion.

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
    ```

7. Save and load the model state.

    Save the dense model weights and dynamic embedding data.
    Use dedicated APIs for sparse weights and optimizer state.
    - Saving: `DynamicEmbDump(save_dir, model, optim=True)`
    - Loading: `DynamicEmbLoad(save_dir, model, optim=True)`

## Model Training Startup

Model training depends on a container environment. You can quickly start the container and train the model using a prebuilt container image. You can also manually install dependencies and deploy the software in the container image environment.

### Option 1: Starting Model Training Using an Existing Container Image

1. Obtain an existing runtime image, start the container, and enter it.
See [OVERVIEW](../../../../docker/OVERVIEW.zh.md) for instructions on starting the container from the prebuilt image and entering it.

2. Configure the environment in the container.

    Run the following commands to configure the environment in the container:

    ```bash
    # Activate the Python virtual environment.
    source /opt/buildtools/torch_v2_pt2.7.1/bin/activate
    # Set the CANN version.
    bash /usr/local/set_cann_env.sh A5
    ```

3. Start model training.

    Go to the sample code directory:

    ```bash
    cd /RecSDK/torch_rec_v2_examples/little_demo
    ```

    Download the MovieLens-1M dataset and extract it into the `ml-1m` folder in the current directory.
    Dataset link: [MovieLens-1M](https://files.grouplens.org/datasets/movielens/ml-1m.zip)

    The downloaded file is named `ml-1m.zip`. Decompress it. By default, the sample model uses `./ml-1m` as the data path. To change the path, specify the `--data_path` parameter in the `run.sh` script.

    Run the model script:

    ```bash
    bash run.sh
    ```

    The `run.sh` script automatically sets `PYTHONPATH` and starts training.
    The `--train` CLI parameter starts model training. The `--dump` and `--load` parameters enable end-to-end verification and run the full process of training, saving, loading, and inference.

    ```bash
    torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:6000 --nnodes=1 --nproc-per-node=1 main.py --train "$@"
    torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:6000 --nnodes=1 --nproc-per-node=1 main.py --load --dump "$@"
    ```

### Option 2: Manually Preparing the Running Environment and Starting Model Training

1. See the [Install Rec SDK Torch](./recsdk_torch_installation_guide.md#section182972951211) section to prepare the container environment, then start and enter the container.
2. Download the [Rec SDK Torch Little Demo Sample](https://gitcode.com/Ascend/RecSDK/tree/develop_examples_and_tools/torch_rec_v2_examples/little_demo) code.
3. Start model training as in option 1.
