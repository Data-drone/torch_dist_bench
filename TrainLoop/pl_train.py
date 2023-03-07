#####
#
# This main loop is intended to be run inside of databricks
# ddp mode isn't supported in interactive mode so is run via %sh
# running via %sh requires that a terminal session is started and databricks cli configured
# running via %sh won't log the code against the notebook

import torch
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule
from typing import Type

# do tensorboard profiling
import torch.profiler
# log hardware stats
from pytorch_lightning.callbacks import DeviceStatsMonitor

# Adding mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

# pytorch profiling
from pytorch_lightning.profiler import PyTorchProfiler
import os

# spark horovod
from sparkdl import HorovodRunner
import horovod.torch as hvd

#

#


def main_hvd(mlflow_db_host:str, mlflow_db_token:str, 
            data_module:Type[LightningDataModule], model:Type[LightningModule], 
            root_dir:str, epochs: int, run_name:str, experiment_path:str, devices=1, 
            strategy='horovod'):

    """
    
    In order to leverage horovod we need to wrap the main train function with a hvd.init
    
    Args:
        mlflow_db_host: the url for your workspace 
        mlflow_db_token: A valid access token for your workspace see: https://docs.databricks.com/dev-tools/api/latest/authentication.html
        data_module: pl LightningDataModule
        model: pl LightningModule
        root_dir: We need to set this to a /dbfs/ folder
        epochs:
        run_name: This name is used for MLFlow and also for the profiler logs
        experiment_id:
        args/kwargs: Args/Kwargs get fed into the pl.Trainer class
    
    """
    import logging
    import horovod.spark.task as tk
    logging.getLogger().setLevel(logging.INFO)

    hvd.init()

    # mlflow workaround for python subprocess not receiving notebook token and workspace url
    mlflow.set_tracking_uri("databricks")
    os.environ['DATABRICKS_HOST'] = mlflow_db_host
    os.environ['DATABRICKS_TOKEN'] = mlflow_db_token

    ## Multi-gpu seems to work funny....
    # manually set the CUDA VISIBLE DEVICES for current config of 4GPU workers
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    os.environ['NCCL_P2P_DISABLE'] = '1'
    #

    logging.info('local_rank: {0}'.format(hvd.local_rank()))
    #available_devices = tk.get_available_devices()
    device_string = ','.join(str(x) for x in tk.get_available_devices())

    #assert type(available_devices) in [list, int]

    return main_train_loop(data_module=data_module, model=model, strategy=strategy, 
                           devices=1, node_id=hvd.rank(),
                            world_size=hvd.size(), root_dir=root_dir, epochs=epochs, run_name=run_name, 
                            experiment_path=experiment_path)


def main_train_loop(data_module:Type[LightningDataModule], model:Type[LightningModule], 
                    strategy:str, root_dir:str, epochs, run_name,
                    experiment_path, devices, node_id, world_size):

    # set saving folders
    log_dir = os.path.join(root_dir, 'logs')

    loggers = []
    tb_logger = pl.loggers.tensorboard.TensorBoardLogger(save_dir=log_dir, name=run_name,log_graph=True)
    mlflow_logger = pl.loggers.MLFlowLogger(experiment_name=experiment_path, run_name=run_name)

    loggers.append(tb_logger)
    loggers.append(mlflow_logger)

    callbacks = []
    device_stats = DeviceStatsMonitor()

    callbacks.append(device_stats)


    # Profilers - This is the key part to make sure that we can log perf stats and debug what is going wrong.
    # See: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html

    profiler = PyTorchProfiler(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=3,
            repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(log_dir,run_name), worker_name='worker'),
        record_shapes=True,
        profile_memory=True,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
        with_stack=True)
    
    # debug logger

    #try:
    #    logging.info('CUDA Devices: {0}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
    #except KeyError:
    #    logging.info('CUDA VISIBLE DEVICES not set')
    #


    # main pytorch lightning trainer
    # we also feed all the args/kwargs in
    # See: https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html
    # GPUs is deprecated but we can still use it in 1.7.7
    trainer = pl.Trainer(
        max_epochs=epochs,
        log_every_n_steps=100,
        accelerator='gpu',
        devices=devices,
        callbacks=callbacks,
        logger=loggers,
        strategy=strategy,
        profiler=profiler,
        default_root_dir=root_dir #otherwise pytorch lightning will write to local
        #profiler=profiler # for tensorboard profiler
    )

    trainer.fit(model, data_module)

    return trainer
