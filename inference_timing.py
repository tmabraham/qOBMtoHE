from fastai.vision.all import *
from upit.models.cyclegan import *
from upit.data.unpaired import *
from upit.train.cyclegan import *
from upit.inference.cyclegan import *
from upit.tracking.wandb import *
import argparse
import wandb
from utils import *

import torchvision
import os
import time

torch.backends.cudnn.benchmark = True


@torch.compile(mode='max-autotune')
def call_model(model,im):
    with torch.inference_mode(): preds = model(im)/2 + 0.5
    return preds

def get_preds_cyclegan(learn,test_path,pred_path,convert_to='B',device='cuda',suffix='tif'):
    if convert_to=='B': model = learn.model.G_B.to(device)
    else:               model = learn.model.G_A.to(device)
    im = Image.open(file)
    im = ((torchvision.transforms.ToTensor()(im)-0.5)/0.5).unsqueeze(0).cuda()
    start_time = time.time()
    preds = call_model(model, im)
    print(f"{time.time()-start_time} seconds")
    torchvision.utils.save_image(preds,pred_path)

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
    if os.path.isfile(args.load_model_name+'.pth'):
        model_dir = Path(args.load_model_name+'.pth').parents[0]
        model_file_name = Path(args.load_model_name+'.pth').stem
    else:
        model_artifact = wandb.run.use_artifact(args.load_model_name+":latest")
        model_dir = Path(model_artifact.download())
        model_file_name = model_dir.ls()[0].name[:-4]
    print(model_file_name)
    learn.model_dir = model_dir
    learn.load(model_file_name)

    import torch._dynamo
    torch._dynamo.reset()

    # Make directory for patches of test images
    test_path = qobm2he_path/args.test_folder
    preds_folder = f'./preds_cyclegan_{args.experiment_name}'
    os.makedirs(preds_folder, exist_ok=True)
    files = [Path(os.path.join(test_path,f)) for f in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, f))]
    for file in files:
        print(file.stem)
        get_preds_cyclegan(learn, file, preds_folder+'/'+file.stem+'_conversion.png')
    
    preds_artifact = wandb.Artifact(f'preds_cyclegan_{args.experiment_name}', type="predictions", description="predictions on train set", metadata=dict(wandb.config))
    preds_artifact.add_dir(preds_folder)

    wandb.finish()
    print('Done!')
