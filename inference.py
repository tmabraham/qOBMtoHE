from fastai.vision.all import *
from upit.models.cyclegan import *
from upit.data.unpaired import *
from upit.train.cyclegan import *
from upit.inference.cyclegan import *
from upit.tracking.wandb import *
import argparse
import wandb
from utils import *

def parse_args():
    parser = argparse.ArgumentParser(description='Inference for qOBM-to-H&E conversion')
    parser.add_argument('--experiment_name', default='cyclegan_inference', type=str, help='Name of the experiment')
    parser.add_argument('--dataset_name', default='tumor_1_60x_complete_v1', type=str, help='Name of the dataset')
    parser.add_argument('--data_dir', default='/mnt/qobmtohe/', type=str, help='Directory with the dataset')
    parser.add_argument('--load_model_name', type=str, required=True, help='Name of the model to load')
    parser.add_argument('--disc_layers', default=3, type=int, help='Number of discriminator layers')
    parser.add_argument('--gen_blocks', default=9, type=int, help='Number of residual blocks in the generator')
    parser.add_argument('--qobm_folder', default='trainA', type=str, help='Folder with qOBM images for training')
    parser.add_argument('--he_folder', default='trainB', type=str, help='Folder with H&E images for training') 
    parser.add_argument('--test_folder', default='test_qobm', type=str, help='Folder with qOBM images for testing')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
    parser.add_argument('--load_size', default=512, type=int, help='Size of images to be loaded')
    parser.add_argument('--crop_size', default=256, type=int, help='Size of images to be loaded')
    parser.add_argument('--montage_size', default=[1848, 1848], nargs=2, type=int, help='Size of test image/montage')
    parser.add_argument('--overlapping_stride', default=167, type=int, help='Stride of overlapping patches')
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
    learn = cycle_learner(dls, 
                          cycle_gan,
                          cbs=[UPITWandbCallback(log_preds=True, 
                                                 log_model=True, 
                                                 log_dataset=qobm2he_path, 
                                                 folder_names=[qobm_path.name, he_path.name]
                                                 )]
                        ) # Set up learner

    # Load model
    model_artifact = wandb.run.use_artifact(args.load_model_name+":latest")    
    model_dir = Path(model_artifact.download())
    model_file_name = model_dir.ls()[0].name[:-4]
    print(model_file_name)
    learn.model_dir = model_dir
    learn.load(model_file_name)

    # Make directory for patches of test images
    test_path = qobm2he_path/args.test_folder
    preds_folder = f'./preds_cyclegan_{args.experiment_name}_stride{args.overlapping_stride}'
    os.makedirs(preds_folder, exist_ok=True)
    files = [Path(os.path.join(test_path,f)) for f in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, f))]
    for file in files: 
        print(file.stem)
        create_tiles(str(file),(args.load_size,args.load_size),args.overlapping_stride,str(test_path/file.stem)+'/')
        get_preds_cyclegan(learn,str(test_path/file.stem),preds_folder+'/'+file.stem,bs=args.batch_size)
        images = files_to_array((preds_folder+'/'+file.stem),window_size=(args.load_size,args.load_size),total_size=tuple(args.montage_size),stride=args.overlapping_stride) # Collate predictions into array
        montage = formMontage(images,stride=args.overlapping_stride) # Create montage of predictions
        plt.imsave(preds_folder+'/_'+file.stem+f'_stride{args.overlapping_stride}_montage.png',montage.astype('uint8'))
    
    preds_artifact = wandb.Artifact(f'preds_cyclegan_{args.experiment_name}_stride{args.overlapping_stride}', type="predictions", description="predictions on train set", metadata=dict(wandb.config))
    preds_artifact.add_dir(preds_folder)

    #wandb.finish()
    print('Done!')
