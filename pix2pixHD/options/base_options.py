import argparse
import os
from util import util
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):    
        # experiment specifics
        self.parser.add_argument('--name', type=str, default='label2city', help='name of the experiment. It decides where to store samples and models')        
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--model', type=str, default='pix2pixHD', help='which model to use')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')        
        self.parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
        self.parser.add_argument('--data_type', default=32, type=int, choices=[8, 16, 32], help="Supported data type i.e. 8, 16, 32 bit")
        self.parser.add_argument('--verbose', action='store_true', default=False, help='toggles verbose')
        self.parser.add_argument('--fp16', action='store_true', default=False, help='train with AMP')
        self.parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')

        # input/output sizes       
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=1024, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=512, help='then crop to this size')
        self.parser.add_argument('--label_nc', type=int, default=35, help='# of input label channels')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

        # for setting inputs
        self.parser.add_argument('--dataroot', type=str, default='./datasets/cityscapes/') 
        self.parser.add_argument('--resize_or_crop', type=str, default='scale_width', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')        
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation') 
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')                
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

        # for displays
        self.parser.add_argument('--display_winsize', type=int, default=512,  help='display window size')
        self.parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')

        # for generator
        self.parser.add_argument('--netG', type=str, default='global', help='selects model to use for netG')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--n_downsample_global', type=int, default=4, help='number of downsampling layers in netG') 
        self.parser.add_argument('--n_blocks_global', type=int, default=9, help='number of residual blocks in the global generator network')
        self.parser.add_argument('--n_blocks_local', type=int, default=3, help='number of residual blocks in the local enhancer network')
        self.parser.add_argument('--n_local_enhancers', type=int, default=1, help='number of local enhancers to use')        
        self.parser.add_argument('--niter_fix_global', type=int, default=0, help='number of epochs that we only train the outmost local enhancer')        

        # for instance-wise features
        self.parser.add_argument('--n_weather_classes', type=int, default=0, help='# of weather conditions (0=disabled, 4=Sunny/Rain/Night/Fog)')
        self.parser.add_argument('--edge_input', action='store_true', help='if specified, append a 1-channel facade-edge map (from {phase}_edge/) to the generator input')
        self.parser.add_argument('--depth_input', action='store_true', help='if specified, append a 1-channel monocular-depth map (from {phase}_depth/) to the generator input')
        self.parser.add_argument('--normal_input', action='store_true', help='if specified, append a 3-channel surface-normal map (from {phase}_normal/) to the generator input')
        # Night needs to be TOLD where the light is. Label/edge/depth/normal are identical at
        # noon and midnight, and the night corpus (real dashcam footage) has streetlights
        # everywhere while CARLA's Town10HD does not -- so wherever CARLA gave no cue the model
        # painted a black void. Same train/inference relationship as the edge channel that
        # already works here: computed from the real photo at train time, from CARLA's night
        # capture at inference.
        self.parser.add_argument('--light_input', action='store_true', help='if specified, append a 1-channel light/emissive map (from {phase}_light/) to the generator input')
        # chroma prior: 3ch heavily-blurred COLOUR (no luminance) over thing classes.
        # Same train/inference contract as the edge and light channels -- from the real photo at
        # training time, from CARLA's render at inference -- so the model is told what colour an
        # object is instead of inventing one per frame.
        self.parser.add_argument('--veg_weight', type=float, default=1.0, help='upweight vegetation in the perceptual loss. Area-weighted losses give a distant tree almost no gradient, so the model never learns it; measured, the render reaches 56%% of the training photographs vegetation detail')
        self.parser.add_argument('--veg_extra', type=float, default=0.0,
                                 help='additive vegetation-only perceptual term. Unlike --veg_weight '
                                      'this does not dilute the rest of the frame: the plain VGG loss '
                                      'is kept at full strength and this is added on top.')
        self.parser.add_argument('--far_boost', type=float, default=0.0, help='extra vegetation weight proportional to distance, using the inverse-depth channel (far = low). Targets the specific failure that render detail DROPS with distance (far/near 0.83) where photos and CARLA hold it (1.05-1.11)')
        self.parser.add_argument('--thing_weight', type=float, default=1.0,
                                 help='per-pixel weight on thing classes in the VGG loss; 1.0 = uniform (original behaviour)')
        self.parser.add_argument('--chroma_input', action='store_true', help='append 3ch chroma-prior channel')
        self.parser.add_argument('--temporal', action='store_true', help='temporal model: condition on previous frame + flow-warp consistency loss (needs sequence data)')
        self.parser.add_argument('--lambda_temporal', type=float, default=10.0, help='weight of the temporal consistency loss')
        self.parser.add_argument('--prev_aug', type=int, default=0, help='randomise the amplitude of the previous-frame channel fed to G during training, so the out-of-distribution inputs it meets at inference (its own output, a scaled version, or a blank frame) are ones it has actually seen')
        self.parser.add_argument('--temporal_tex_thresh', type=float, default=0.02, help='min local gradient energy for a pixel to count in the temporal loss; flow is meaningless in textureless regions and matching a bad warp there bakes ghost edges into the weights')
        self.parser.add_argument('--video_disc', action='store_true', help='vid2vid: add a temporal (video) discriminator over frame pairs (prev,cur) that forces sharp temporal consistency (fixes the temporal-only blur). Requires --temporal.')
        self.parser.add_argument('--lambda_vgan', type=float, default=1.0, help='weight of the video-discriminator adversarial loss on G')
        self.parser.add_argument('--multiframe', action='store_true', help='non-autoregressive temporal: stack label+edge of frames t-1,t,t+1 as G input, predict only t. Needs multiframe sequence data. Cannot blur (never feeds generated output back).')
        self.parser.add_argument('--mf_window', type=int, default=1, help='multiframe temporal window: +-W neighbour frames stacked (1 = 3 frames t-1,t,t+1)')
        self.parser.add_argument('--mf_align', action='store_true', help='multiframe: optical-flow-warp neighbour labels/edges onto the current frame before stacking (fixes v29 misalignment). Needs {phase}_img for flow.')
        self.parser.add_argument('--no_instance', action='store_true', help='if specified, do *not* add instance map as input')
        self.parser.add_argument('--instance_feat', action='store_true', help='if specified, add encoded instance features as input')
        self.parser.add_argument('--label_feat', action='store_true', help='if specified, add encoded label features as input')        
        self.parser.add_argument('--feat_num', type=int, default=3, help='vector length for encoded features')        
        self.parser.add_argument('--load_features', action='store_true', help='if specified, load precomputed feature maps')
        self.parser.add_argument('--n_downsample_E', type=int, default=4, help='# of downsampling layers in encoder') 
        self.parser.add_argument('--nef', type=int, default=16, help='# of encoder filters in the first conv layer')        
        self.parser.add_argument('--n_clusters', type=int, default=10, help='number of clusters for features')        

        self.initialized = True

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        
        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk        
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        if save and not self.opt.continue_train:
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt
