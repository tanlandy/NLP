# Multiple_GPUs

## 选型

单GPU可以放得下完整模型：

- DPP(Distributed DataParallel)
- ZeRO

单GPU放不下：

- PP(PipelineParallel)
- ZeRO
- TP(TensorParallel)

## DPP(Distributed DataParallel)

DDP:

- At the start time the main process replicates the model once from GPU 0 to the rest of GPUs
- Then for each batch:
  - Each GPU directly consumes its mini-batch of data.
  - During backward, once the local gradients are ready, they are averaged across all processes.

DP:

For each batch:

- GPU 0 reads the batch of data and then sends a mini-batch to each GPU.
- The up-to-date model is replicated from GPU 0 to each GPU.
- forward is executed, and output from each GPU is sent to GPU 0 to compute the loss.
- The loss is distributed from GPU 0 to all GPUs, and backward is run.
- Gradients from each GPU are sent to GPU 0 and averaged.

## ZeRO

[Installation](https://www.deepspeed.ai/tutorials/advanced-install/)

[Getting Started](https://www.deepspeed.ai/getting-started/)
[megatron](https://www.deepspeed.ai/tutorials/megatron/)
[zero](https://www.deepspeed.ai/tutorials/zero/)
[HF](https://huggingface.co/docs/transformers/main_classes/deepspeed)
[HF transformers](https://huggingface.co/docs/transformers/deepspeed)
[DP Features](https://www.deepspeed.ai/training/)

ZeRO:
The difference is that instead of replicating the full model parameters, gradients and optimizer states, each GPU stores only a slice of it. Then, at run-time when the full layer parameters are needed just for the given
layer, all GPUs synchronize to give each other parts that they miss.
