_base_ = './vit-large-p16_8xb128-coslr-50e_in1k.py'


runner_type = 'FlexibleRunner'
strategy = dict(
    type='FSDPStrategy',
    model_wrapper=dict(auto_wrap_policy=dict(type='transformer_encoder_wrap_policy')))

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=20)
)

custom_hooks = [
    dict(type='ProfilerHook',
         by_epoch=False,
         activity_with_cpu=True,
         activity_with_cuda=True,
         schedule=dict(wait=1, warmup=2, active=5),
         on_trace_ready=dict(type='tb_trace'),
         with_stack=True,
         profile_memory=True,
         profile_times=10)]

train_dataloader = dict(batch_size=128)
