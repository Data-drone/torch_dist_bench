# Training Script intended to be run as a file

from TrainLoop.pl_train import main_train_loop
from Models.resnet_model import ResnetClassification
from DataLoaders.imagenette_file import ImagenetteDataModule

import argparse
import os
import mlflow

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data-path', dest='data_path', required=True)
    parser.add_argument('--root-folder', dest='root_folder', required=True)
    parser.add_argument('--experiment-path', dest='experiment_path', required=True)
    parser.add_argument('--mlflow-host', dest='db_host', required=True)
    parser.add_argument('--mlflow-token', dest='db_token', required=True)
    
    parser.add_argument('--num-gpus', dest='num_gpus', required=False)
    parser.add_argument('--num-epochs', dest='num_epochs', required=False)
    parser.add_argument('--prefix', dest='prefix', required=False)

    args = parser.parse_args()

    os.environ['DATABRICKS_HOST'] = args.db_host
    os.environ['DATABRICKS_TOKEN'] = args.db_token

    #mlflow.set_experiment(experiment_id=args.experiment_id)
    # NCCL P2P can cause issues with incorrect peer settings
    ## Turn this off to scale for now
    os.environ['NCCL_P2P_DISABLE'] = '1'



    # args

    num_workers = 8
    pin_memory = True
    batch_size = 64
    total_epochs = int(args.num_epochs)
    strategy = 'ddp'
    tot_gpus = int(args.num_gpus)

    if args.prefix:
        run_name = f'distributor_run_{args.prefix}_{tot_gpus}_gpu'
    else:
        run_name = f'distributor_run_{tot_gpus}_gpu'

    # end args

    data_module = ImagenetteDataModule(data_dir=args.data_path, batch_size=batch_size, 
                                   num_workers=num_workers, pin_memory=pin_memory)

    model = ResnetClassification(*data_module.image_shape, num_classes=data_module.num_classes)

    main_train_loop(data_module, model, 
           num_devices=tot_gpus, root_dir=args.root_folder, 
           epochs=total_epochs, strategy=strategy, 
           experiment_path=args.experiment_path, 
           run_name=run_name,
           node_id=0, world_size=1)

