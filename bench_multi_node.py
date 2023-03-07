# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Multi-Node
#

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Setup

# COMMAND ----------

from sparkdl import HorovodRunner
import horovod.spark
import os

# COMMAND ---------

from TrainLoop.pl_train import main_hvd, main_train_loop
from Models.resnet_model import ResnetClassification
from DataLoaders.imagenette_file import ImagenetteDataModule

# COMMAND ---------

username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

databricks_host = dbutils.secrets.get(scope="scaling_dl", key="host_workspace")
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

os.environ['DATABRICKS_HOST'] = databricks_host
os.environ['DATABRICKS_TOKEN'] = databricks_token

imagenette_data_path = f'/dbfs/Users/{username}/data/imagenette2'

experiment_id = 2324556583979176
experiment_path = f'/Users/{username}/brian_benchmark_exp'

root_folder = f'/dbfs/Users/{username}/benchmark_exp'

total_epochs = 15
num_workers = 6
pin_memory = True
batch_size = 64

data_module = ImagenetteDataModule(data_dir=imagenette_data_path, batch_size=batch_size, 
                                   num_workers=num_workers, pin_memory=pin_memory)

model = ResnetClassification(*data_module.image_shape, num_classes=data_module.num_classes)



# COMMAND ---------

# MAGIC %md
# MAGIC 
# MAGIC ## Horovod

# COMMAND ---------

# hr = HorovodRunner(np=2)
# hr.run(main_hvd, 
#          mlflow_db_host=databricks_host, 
#          mlflow_db_token=databricks_token, 
#          data_module=data_module, 
#          model=model, 
#          root_dir=root_folder, 
#          epochs=total_epochs, 
#          run_name='hvd_runner_dual_multi', 
#          experiment_path=experiment_path)

# COMMAND ---------

np = 2
horovod.spark.run(main_hvd, kwargs={
                 'mlflow_db_host': databricks_host, 
                 'mlflow_db_token': databricks_token, 
                 'data_module': data_module, 
                 'model': model, 
                 'root_dir': root_folder, 
                 'epochs': total_epochs, 
                 'run_name': 'hvd_runner_multi_dual', 
                 'experiment_path': experiment_path},
                  num_proc = np,
                  verbose=2)

# COMMAND ----------

np = 4
horovod.spark.run(main_hvd, kwargs={
                 'mlflow_db_host': databricks_host, 
                 'mlflow_db_token': databricks_token, 
                 'data_module': data_module, 
                 'model': model, 
                 'root_dir': root_folder, 
                 'epochs': total_epochs, 
                 'run_name': 'hvd_runner_multi_quad', 
                 'experiment_path': experiment_path},
                  num_proc = np,
                  verbose=2)


# COMMAND ---------

np = 8
horovod.spark.run(main_hvd, kwargs={
                 'mlflow_db_host': databricks_host, 
                 'mlflow_db_token': databricks_token, 
                 'data_module': data_module, 
                 'model': model, 
                 'root_dir': root_folder, 
                 'epochs': total_epochs, 
                 'run_name': 'hvd_runner_multi_octo', 
                 'experiment_path': experiment_path},
                  num_proc = np,
                  verbose=2)

# COMMAND ---------

# MAGIC %md
# MAGIC 
# MAGIC ## Torch Distributor

# COMMAND ---------


path = 'pytorch_lightning_train_script.py'
tot_gpus = 2

TorchDistributor(num_processes=tot_gpus, local_mode=False, use_gpu=True) \
  .run(path, '--data-path={0}'.format(imagenette_data_path),
       '--root-folder={0}'.format(root_folder),
       '--experiment-path={0}'.format(experiment_path),
      '--mlflow-host={0}'.format(databricks_host),
      '--mlflow-token={0}'.format(databricks_token),
      '--num-gpus={0}'.format(tot_gpus),
      '--num-epochs={0}'.format(total_epochs))

# COMMAND ---------

# Four GPU

path = 'pytorch_lightning_train_script.py'
tot_gpus = 4

TorchDistributor(num_processes=tot_gpus, local_mode=False, use_gpu=True) \
  .run(path, '--data-path={0}'.format(imagenette_data_path),
       '--root-folder={0}'.format(root_folder),
       '--experiment-path={0}'.format(experiment_path),
      '--mlflow-host={0}'.format(databricks_host),
      '--mlflow-token={0}'.format(databricks_token),
      '--num-gpus={0}'.format(tot_gpus),
      '--num-epochs={0}'.format(total_epochs))

# COMMAND ---------


path = 'pytorch_lightning_train_script.py'
tot_gpus = 8

TorchDistributor(num_processes=tot_gpus, local_mode=False, use_gpu=True) \
  .run(path, '--data-path={0}'.format(imagenette_data_path),
       '--root-folder={0}'.format(root_folder),
       '--experiment-path={0}'.format(experiment_path),
      '--mlflow-host={0}'.format(databricks_host),
      '--mlflow-token={0}'.format(databricks_token),
      '--num-gpus={0}'.format(tot_gpus),
      '--num-epochs={0}'.format(total_epochs))

# COMMAND ---------
