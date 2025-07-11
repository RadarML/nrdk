# Neural Radar Development Kit API

## Configuration with Hydra

## Code Examples

!!! info

    These examples use the configuration files included with the [GRT Reference Implementation](../grt/index.md); to use these examples "out of the box," make sure your working directory is in the `nrdk` directory (and that you have the [GRT reference implementation extras](../grt/index.md) installed).

??? example "Minimal training code"

    ```python
    import hydra
    import lightning as L

    from nrdk.framework import ADLLightningModule, TensorBoardLogger

    with hydra.initialize(version_base=None, config_path="./grt/config"):
        cfg = hydra.compose(config_name="main", overrides=[
            "+sensors=[lidar3d]", "+objectives=[lidar3d]", "+decoders=[lidar3d]"])

    inst = hydra.utils.instantiate(cfg)
    datamodule = inst["datamodule"](transforms=inst["transforms"])
    lightningmodule = ADLLightningModule(
        model=inst["model"],
        objective=inst["objectives"],
        optimizer=inst["optimizer"],
        transforms=inst["transforms"],
        vis_interval=1, vis_samples=16)

    logger = TensorBoardLogger(save_dir="./tmp")
    trainer = L.Trainer(precision="16-mixed", logger=logger)
    trainer.fit(model=lightningmodule, datamodule=datamodule)
    ```

??? example "Run and visualize a sample batch"

    ```python
    import hydra
    import optree
    import lightning as L
    from matplotlib import pyplot as plt

    with hydra.initialize(version_base=None, config_path="./grt/config"):
        cfg = hydra.compose(config_name="main", overrides=[
            "+sensors=[lidar3d,lidar2d,semseg,vel]",
            "+objectives=[lidar3d,lidar2d,semseg,vel]",
            "+decoders=[lidar3d,lidar2d,semseg,vel]"])

    inst = hydra.utils.instantiate(cfg)
    datamodule = inst["datamodule"](transforms=inst["transforms"])

    train = datamodule.train_dataloader()
    for batch in train:
        batch = datamodule.transforms.batch(batch)
        break

    objectives = inst["objectives"]
    model = inst["model"].cuda()

    batch_gpu = optree.tree_map(lambda x: x.cuda(), batch)
    output = model(batch_gpu)

    metrics = objectives(batch_gpu, output, train=True)
    print(metrics)

    visualizations = objectives.visualizations(batch_gpu, output)
    fig, axs = plt.subplots(4, 1, figsize=(20, 20))
    for ax, (k, v) in zip(axs, visualizations.items()):
        ax.imshow(v)
    ```
