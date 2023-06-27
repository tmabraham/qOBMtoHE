from fastai.vision.all import *
from upit.models.cyclegan import *
from upit.data.unpaired import *
from upit.train.cyclegan import *
from upit.inference.cyclegan import *
from upit.tracking.wandb import *
import argparse
import wandb

def parse_args():
    parser = argparse.ArgumentParser(description='Training CycleGAN for qOBM-to-H&E conversion')
    parser.add_argument('--experiment_name', default='cyclegan_experiment', type=str, help='Name of the experiment')
    parser.add_argument('--dataset_name', default='tumor_1_60x_complete_v1', type=str, help='Name of the dataset')
    parser.add_argument('--load_model_name', type=str, help='Name of the model to load')
    parser.add_argument('--data_dir', default='/mnt/qobmtohe/', type=str, help='Directory with the dataset')
    parser.add_argument('--qobm_folder', default='trainA', type=str, help='Folder with qOBM images for training')
    parser.add_argument('--he_folder', default='trainB', type=str, help='Folder with H&E images for training') 
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size')
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs (half constant, other half linear decay)')
    parser.add_argument('--load_size', default=512, type=int, help='Size of images to be loaded')
    parser.add_argument('--crop_size', default=256, type=int, help='Size of images to be loaded')
    parser.add_argument('--disc_layers', default=3, type=int, help='Number of discriminator layers')
    parser.add_argument('--gen_blocks', default=9, type=int, help='Number of residual blocks in the generator')
    parser.add_argument('--lambda_A', default=10.0, type=float, help='Weight for cycle loss (A -> B -> A)')
    parser.add_argument('--lambda_B', default=10.0, type=float, help='Weight for cycle loss (B -> A -> B)')
    parser.add_argument('--lambda_identity', default=0.5, type=float, help='Weight for identity loss')
    parser.add_argument('--lr', default=2e-4, type=float, help='Learning rate')
    parser.add_argument('--gpu', default=-1, type=int, help='GPU ID')
    parser.add_argument('--wandb_project', type=str, default='qOBM-to-H&E-final', help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default='tmabraham', help='Wandb entity name')
    args = parser.parse_args()
    return args

if __name__ == '__main__': 
    args = parse_args()
    wandb.init(project=args.wandb_project, entity=args.wandb_entity)
    wandb.run.name = args.experiment_name
    wandb.config.update(args)
    if args.gpu != -1:
        torch.cuda.set_device(args.gpu) #set GPU id
    # Setup data and model
    dataset_path = Path(args.data_dir)
    qobm2he_path = dataset_path/args.dataset_name
    print(qobm2he_path)
    qobm_path = qobm2he_path/args.qobm_folder
    he_path = qobm2he_path/args.he_folder
    try:
        dls = get_dls(qobm_path, he_path, bs=args.batch_size, load_size=args.load_size, crop_size=args.crop_size)
    except: 
        print('There was and error loading the dataset and setting up the data loaders.')
        sys.exit(1)
    
    cycle_gan = CycleGAN(3,3, disc_layers=args.disc_layers, gen_blocks=args.gen_blocks)
    loss_func = partial(CycleGANLoss, l_A=args.lambda_A, l_B=args.lambda_B, l_idt=args.lambda_identity)
    learn = cycle_learner(dls, 
                          cycle_gan,
                          loss_func=loss_func, 
                          opt_func=partial(Adam,mom=0.5,sqr_mom=0.999), 
                          show_imgs=False, 
                          cbs=[UPITWandbCallback(log_preds=True, 
                                                 log_model=True, 
                                                 log_dataset=qobm2he_path, 
                                                 folder_names=[qobm_path.name, he_path.name]
                                                 )]
                        ) # Set up learner
    
    
    # Load model
    if args.load_model_name:
        model_artifact = wandb.run.use_artifact(args.load_model_name+":latest")    
        model_dir = Path(model_artifact.download())
        model_file_name = model_dir.ls()[0].name[:-4]
        print(model_file_name)
        learn.model_dir = model_dir
        learn.load(model_file_name)

    if args.epochs: learn.fit_flat_lin(int(args.epochs/2),int(args.epochs/2), args.lr) # Training
    
    model_file_name = f'{args.epochs}fit_cyclegan_{args.experiment_name}'
    learn.save(model_file_name) # Save model
    # save trained model as artifact
    trained_model_artifact = wandb.Artifact(
                model_file_name, type="model",
                description="trained CycleGAN",
                metadata=dict(wandb.config))
    trained_model_artifact.add_file(str(learn.path/learn.model_dir/model_file_name)+'.pth')
    wandb.run.log_artifact(trained_model_artifact)

    print('Done!')
