# Software Architecture

The NRDK software architecture is designed around modular components which can be assembled by dependency injection via hydra.

In addition to the core framework and specification, the `nrdk` library hosts common, stable implementations of each component; experimental "research" components should be implemented in separate repositories, and only merged once stable and ready to publish.

=== "NRDK Architecture"

    ![nrdk abstract architecture](assets/nrdk_abstract.svg)

=== "GRT Reference Implementation"

    ![nrdk grt reference implementation](assets/nrdk_grt.svg)

## Core Framework

The NRDK framework is based on Pytorch Lightning, and centered around a core [`NRDKLightningModule`][nrdk.framework.NRDKLightningModule].

!!! tip

    The core framework can be completely bypassed if you wish; simply use the data loading, preprocessing, and training objectives without using the high level control flow provided in `nrdk.framework`.

- A model is a [`nn.Module`][torch.nn.Module] which takes each batch as an input, and returns the model's predictions.

    !!! info
    
        We provide a [`TokenizerEncoderDecoder`][nrdk.framework.TokenizerEncoderDecoder] module, which wraps a tokenizer, encoder, and one or more decoders into a standardized model container.

- The model may or may not have GPU-side preprocessing which is expected to be performed on each batch.
- The model is trained to fit an [`Objective`][abstract_dataloader.ext.objective.Objective], which takes the batch and the predictions as input and returns the loss.

## Data Loading

The NRDK leans heavily on the core [abstract dataloader specifications](https://wiselabcmu.github.io/abstract-dataloader/), which describe a set of modular and composable data loading &mdash;
> [`Sensor`][abstract_dataloader.spec.Sensor] &#8594; [`Trace`][abstract_dataloader.spec.Trace] &#8594; [`Dataset`][abstract_dataloader.spec.Dataset]

&mdash; and preprocessing &mdash;
> [`Pipeline`][abstract_dataloader.spec.Pipeline] := [`.sample:Transform`][abstract_dataloader.spec.Transform] &#8594; [`Collate`][abstract_dataloader.spec.Collate] &#8594; [`.batch:Transform`][abstract_dataloader.spec.Transform]

&mdash; component interfaces.

- When using data collected by [`red-rover`](https://radarml.github.io/red-rover/) and stored in the [`roverd`](https://radarml.github.io/red-rover/roverd/) format, read data using the [`red-rover/roverd`](https://radarml.github.io/red-rover/roverd/) library, which provides `abstract-dataloader` compliant [`Dataset`][roverd.Dataset], [`Trace`][roverd.Trace], and [`Sensor`][roverd.sensors] implementations.

- To preprocess this data during training, we also supply the [`nrdk.roverd`](nrdk/roverd.md) submodule, which implements data preprocessing for each `red-rover` data modality; these are intended to be combined into a [`Transform`][abstract_dataloader.spec.Transform] using [`abstract_dataloader.ext.graph.Transform`][abstract_dataloader.ext.graph.Transform].

    !!! tip

        While nominally designed for `red-rover` data, the `nrdk.roverd` transforms are implemented using [protocol types](https://peps.python.org/pep-0544/) which allow any compatible types to be substituted as inputs.

- Data augmentations can also be specified via the [`abstract_dataloader.ext.augment`][abstract_dataloader.ext.augment] data augmentation specifications.

## Training Objectives

Training objectives for the NRDK use the [`abstract_dataloader.ext.objective`][abstract_dataloader.ext.objective] specification. From the abstract dataloader documentation:

- An [`Objective`][abstract_dataloader.ext.objective.Objective] is a callable which returns a (batched) scalar loss
    and a dictionary of metrics.
- Objectives can be combined into a higher-order objective,
    [`MultiObjective`][abstract_dataloader.ext.objective.MultiObjective], which combines their losses and aggregates their
    metrics; specify these objectives using a [`MultiObjectiveSpec`][abstract_dataloader.ext.objective.MultiObjectiveSpec].

We currently provide the following objectives:

| Objective | Ground Truth Type | Model Output Type |
| --------- | ----------------- | ----------------- |
| [`Occupancy2D`][nrdk.objectives.Occupancy2D] | [`Occupancy2DData`][nrdk.objectives.Occupancy2DData] | `Float[Tensor, 'batch t azimuth range']` |
| [`Occupancy3D`][nrdk.objectives.Occupancy3D] | [`Occupancy3DData`][nrdk.objectives.Occupancy3DData] | `Float[Tensor, 'batch t elevation azimuth range']` |
| [`Semseg`][nrdk.objectives.Semseg] | [`SemsegData`][nrdk.objectives.SemsegData] | `Float[Tensor, 'batch t h w cls']` |
| [`Velocity`][nrdk.objectives.Velocity] | [`VelocityData`][nrdk.objectives.VelocityData] | `Float[Tensor, 'batch t 4']` |

!!! tip

    In addition to implementing the abstract specification, each provided objective includes a type specification for their expected model predictions and ground truth data. Implementations using these objectives out-of-the-box only need to provide data which fits these type interfaces.


## Other Modules

The `nrdk` includes a number of other submodules intended as reusable libraries which are not associated with a well-defined abstract specification.

<div class="grid cards" markdown>

- [`nrdk.config`](nrdk/config.md)

    ---

    configuration utilities

- [`nrdk.metrics`](nrdk/metrics.md)

    ---

    training and evaluation metrics

- [`nrdk.modules`](nrdk/modules.md)

    ---

    pytorch [`nn.Module`][torch.nn.Module] implementations outside of the standard library

- [`nrdk.visualization`](nrdk/vis.md)

    ---

    visualizations utilities (e.g., for in-training visualizations)

</div>
