import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

# Mapillary-65 thing classes: riders, person, and the vehicle group. Same list
# gen_chroma_maps.py uses, so the two stay consistent.
MAPILLARY_THING_IDS = [19, 20, 21, 22, 54, 55, 56, 57, 58, 61]
# Vegetation and other self-textured "stuff" the generator has to invent rather than copy.
MAPILLARY_VEG_IDS = [30]


class Pix2PixHDModel(BaseModel):
    def name(self):
        return 'Pix2PixHDModel'
    
    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss, use_temporal=False, use_video=False):
        flags = (True, use_gan_feat_loss, use_vgg_loss, True, True, use_temporal, use_video, use_video, use_video)
        def loss_filter(g_gan, g_gan_feat, g_vgg, d_real, d_fake, g_temporal=0, g_vgan=0, d_v_real=0, d_v_fake=0):
            return [l for (l,f) in zip((g_gan,g_gan_feat,g_vgg,d_real,d_fake,g_temporal,g_vgan,d_v_real,d_v_fake),flags) if f]
        return loss_filter

    def warp(self, img, flow):
        # backward-warp img by flow (cur->prev displacement), differentiable
        import torch.nn.functional as F
        B, C, H, W = img.size()
        yy, xx = torch.meshgrid(torch.arange(H, device=img.device), torch.arange(W, device=img.device))
        base = torch.stack((xx, yy), 0).float().unsqueeze(0)           # 1,2,H,W
        grid = base + flow                                            # B,2,H,W
        gx = 2.0 * grid[:, 0] / max(W - 1, 1) - 1.0
        gy = 2.0 * grid[:, 1] / max(H - 1, 1) - 1.0
        g = torch.stack((gx, gy), dim=3)                              # B,H,W,2
        return F.grid_sample(img, g, align_corners=True, padding_mode='border')
    
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain: # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        self.use_features = opt.instance_feat or opt.label_feat
        self.gen_features = self.use_features and not self.opt.load_features
        input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc

        ##### define networks
        # Generator network
        netG_input_nc = input_nc + getattr(opt, 'n_weather_classes', 0)
        if getattr(opt, 'edge_input', False):
            netG_input_nc += 1
        if getattr(opt, 'depth_input', False):
            netG_input_nc += 1
        if getattr(opt, 'light_input', False):
            netG_input_nc += 1
        if getattr(opt, 'chroma_input', False):
            netG_input_nc += 3
        if getattr(opt, 'normal_input', False):
            netG_input_nc += 3
        if getattr(opt, 'temporal', False):
            netG_input_nc += opt.output_nc   # previous frame (3ch) concatenated for G only
        if getattr(opt, 'multiframe', False):
            # non-autoregressive temporal: G sees label(+edge) of 2*mf_window+1 frames; D still sees 1 frame.
            # per-frame block = current netG_input_nc so far (label one-hot [+edge]); replicate across frames.
            self.mf_nframes = 2 * getattr(opt, 'mf_window', 1) + 1
            netG_input_nc = netG_input_nc * self.mf_nframes
        if not opt.no_instance:
            netG_input_nc += 1
        if self.use_features:
            netG_input_nc += opt.feat_num
        self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers,
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)

        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = input_nc + getattr(opt, 'n_weather_classes', 0) + opt.output_nc
            if getattr(opt, 'edge_input', False):
                netD_input_nc += 1
            if getattr(opt, 'depth_input', False):
                netD_input_nc += 1
            if getattr(opt, 'light_input', False):
                netD_input_nc += 1
            if getattr(opt, 'chroma_input', False):
                netD_input_nc += 3
            if getattr(opt, 'normal_input', False):
                netD_input_nc += 3
            if not opt.no_instance:
                netD_input_nc += 1
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid,
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
            # video (temporal) discriminator: judges frame PAIRS (prev, cur) = 2*output_nc channels
            if getattr(opt, 'video_disc', False):
                self.netD_V = networks.define_D(2 * opt.output_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid,
                                                opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)

        ### Encoder network
        if self.gen_features:          
            self.netE = networks.define_G(opt.output_nc, opt.feat_num, opt.nef, 'encoder', 
                                          opt.n_downsample_E, norm=opt.norm, gpu_ids=self.gpu_ids)  
        if self.opt.verbose:
                print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)
                # D_V only exists once vid2vid has trained; on fresh fine-tune from v21 it is new -> load only when resuming
                if getattr(opt, 'video_disc', False) and opt.continue_train:
                    self.load_network(self.netD_V, 'D_V', opt.which_epoch, pretrained_path)
            if self.gen_features:
                self.load_network(self.netE, 'E', opt.which_epoch, pretrained_path)              

        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss, getattr(opt, 'temporal', False), getattr(opt, 'video_disc', False))
            
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)   
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:             
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)
                
        
            # Names so we can breakout loss
            self.loss_names = self.loss_filter('G_GAN','G_GAN_Feat','G_VGG','D_real', 'D_fake', 'G_Temporal', 'G_VGAN', 'D_V_real', 'D_V_fake')

            # initialize optimizers
            # optimizer G
            if opt.niter_fix_global > 0:                
                import sys
                if sys.version_info >= (3,0):
                    finetune_list = set()
                else:
                    from sets import Set
                    finetune_list = Set()

                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():       
                    if key.startswith('model' + str(opt.n_local_enhancers)):                    
                        params += [value]
                        finetune_list.add(key.split('.')[0])  
                print('------------- Only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)
                print('The layers that are finetuned are ', sorted(finetune_list))                         
            else:
                params = list(self.netG.parameters())
            if self.gen_features:              
                params += list(self.netE.parameters())         
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))                            

            # optimizer D
            params = list(self.netD.parameters())
            if getattr(opt, 'video_disc', False):
                params += list(self.netD_V.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

    def encode_input(self, label_map, inst_map=None, real_image=None, feat_map=None, infer=False, weather_id=None, edge_map=None, depth_map=None, normal_map=None, light_map=None, chroma_map=None):
        if self.opt.label_nc == 0:
            input_label = label_map.data.cuda()
        else:
            # create one-hot vector for label map
            size = label_map.size()
            oneHot_size = (size[0], self.opt.label_nc, size[2], size[3])
            input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
            input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)
            if self.opt.data_type == 16:
                input_label = input_label.half()

        # append weather one-hot channels
        n_weather = getattr(self.opt, 'n_weather_classes', 0)
        if n_weather > 0 and weather_id is not None:
            bs, _, h, w = input_label.size()
            weather_map = torch.zeros(bs, n_weather, h, w, device=input_label.device,
                                      dtype=input_label.dtype)
            for b in range(bs):
                wid = int(weather_id[b].item()) % n_weather
                weather_map[b, wid] = 1.0
            input_label = torch.cat([input_label, weather_map], dim=1)

        # append facade-edge channel (1ch, values in [0,1])
        if getattr(self.opt, 'edge_input', False) and edge_map is not None:
            edge_in = edge_map.data.cuda().float()
            if edge_in.dim() == 3:
                edge_in = edge_in.unsqueeze(1)
            input_label = torch.cat([input_label, edge_in.to(input_label.dtype)], dim=1)

        # append monocular-depth channel (1ch, values in [0,1])
        if getattr(self.opt, 'depth_input', False) and depth_map is not None:
            depth_in = depth_map.data.cuda().float()
            if depth_in.dim() == 3:
                depth_in = depth_in.unsqueeze(1)
            input_label = torch.cat([input_label, depth_in.to(input_label.dtype)], dim=1)
        # append light/emissive channel (1ch, values in [0,1])
        if getattr(self.opt, 'light_input', False) and light_map is not None:
            light_in = light_map.data.cuda().float()
            if light_in.dim() == 3:
                light_in = light_in.unsqueeze(1)
            input_label = torch.cat([input_label, light_in.to(input_label.dtype)], dim=1)
        # append chroma prior (3ch in [0,1]) -- colour only, luminance already deliberately flat
        if getattr(self.opt, 'chroma_input', False) and chroma_map is not None:
            chroma_in = chroma_map.data.cuda().float()
            if chroma_in.dim() == 3:
                chroma_in = chroma_in.unsqueeze(0)
            input_label = torch.cat([input_label, chroma_in.to(input_label.dtype)], dim=1)

        # append surface-normal channels (3ch, values in [0,1])
        if getattr(self.opt, 'normal_input', False) and normal_map is not None:
            normal_in = normal_map.data.cuda().float()
            if normal_in.dim() == 3:
                normal_in = normal_in.unsqueeze(0)
            input_label = torch.cat([input_label, normal_in.to(input_label.dtype)], dim=1)

        # get edges from instance map
        if not self.opt.no_instance:
            inst_map = inst_map.data.cuda()
            edge_map = self.get_edges(inst_map)
            input_label = torch.cat((input_label, edge_map), dim=1)         
        input_label = Variable(input_label, volatile=infer)

        # real images for training
        if real_image is not None:
            real_image = Variable(real_image.data.cuda())

        # instance map for feature encoding
        if self.use_features:
            # get precomputed feature maps
            if self.opt.load_features:
                feat_map = Variable(feat_map.data.cuda())
            if self.opt.label_feat:
                inst_map = label_map.cuda()

        return input_label, inst_map, real_image, feat_map

    def encode_frame(self, label_map, edge_map=None):
        """Encode a single frame's label(+edge) into the generator's per-frame input block
        (one-hot label [+ 1ch edge]). Used to stack neighbour frames for --multiframe."""
        size = label_map.size()
        oneHot_size = (size[0], self.opt.label_nc, size[2], size[3])
        enc = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
        enc = enc.scatter_(1, label_map.data.long().cuda(), 1.0)
        if getattr(self.opt, 'edge_input', False) and edge_map is not None and torch.is_tensor(edge_map):
            e = edge_map.data.cuda().float()
            if e.dim() == 3:
                e = e.unsqueeze(1)
            enc = torch.cat([enc, e.to(enc.dtype)], dim=1)
        return enc

    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:            
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def forward(self, label, inst, image, feat, infer=False, weather=None, edge=None, prev=None, flow=None,
                mf_labels=None, mf_edges=None, depth=None, normal=None, light=None, chroma=None):
        # Encode Inputs
        input_label, inst_map, real_image, feat_map = self.encode_input(label, inst, image, feat, weather_id=weather, edge_map=edge, depth_map=depth, normal_map=normal, light_map=light, chroma_map=chroma)

        # Fake Generation
        if self.use_features:
            if not self.opt.load_features:
                feat_map = self.netE.forward(real_image, inst_map)
            input_concat = torch.cat((input_label, feat_map), dim=1)
        else:
            input_concat = input_label
        # non-autoregressive temporal: stack neighbour frames' label(+edge) as G input (D still sees current only)
        if getattr(self.opt, 'multiframe', False) and mf_labels is not None:
            blocks = []
            for k in range(len(mf_labels)):
                ek = mf_edges[k] if (mf_edges is not None and k < len(mf_edges)) else None
                blocks.append(input_label if mf_labels[k] is None else self.encode_frame(mf_labels[k], ek))
            # PAD NEIGHBOURS TO THE CURRENT FRAME'S WIDTH. encode_frame emits label one-hot plus
            # edge only, while the current frame's block also carries depth, normal and chroma --
            # 66 channels against 73 with this project's conditioning stack. But netG_input_nc is
            # computed as (per-frame width) * n_frames, i.e. 219, so the concat would be 205 and
            # silently disagree with the first conv. Zero-pad each neighbour to the full width:
            # the missing channels contribute nothing, the arithmetic matches, and no neighbour
            # depth/normal/chroma has to be generated to use this mode.
            wide = input_concat.size(1)
            for k, b in enumerate(blocks):
                if b.size(1) < wide:
                    pad = torch.zeros(b.size(0), wide - b.size(1), b.size(2), b.size(3),
                                      device=b.device, dtype=b.dtype)
                    blocks[k] = torch.cat([b, pad], dim=1)
            input_concat = torch.cat(blocks, dim=1)
        # temporal: condition on previous frame (concatenated for G only)
        elif getattr(self.opt, 'temporal', False) and prev is not None:
            prev = prev.data.cuda().to(input_concat.dtype)
            prev_in = prev
            # PREV-CHANNEL AUGMENTATION. Training only ever showed G a pristine REAL previous
            # frame, but at inference every available option is out of distribution: its own
            # output (which compounds into a crosshatch), a scaled-down version of it, or a blank
            # frame. Measured consequence: a faint ghost of building edges in the sky, ~7x the
            # parent, which survived removing the video discriminator (7.1x) AND texture-gating
            # the temporal loss (6.7x) -- so it is neither of those terms, it is the input itself.
            # Randomising the amplitude here makes the inference-time inputs things G has
            # actually seen. Zero is included explicitly because a blank prev is what frame 0 gets.
            # NOTE this perturbs ONLY the copy fed to G; `prev` itself stays pristine because it
            # is the target of the temporal loss and the video discriminator below.
            if self.isTrain and getattr(self.opt, 'prev_aug', 0):
                r = torch.rand(1).item()
                sc = 0.0 if r < 0.2 else (0.2 + 0.8 * torch.rand(1).item())
                prev_in = prev * sc
            input_concat = torch.cat((input_concat, prev_in), dim=1)
        fake_image = self.netG.forward(input_concat)

        # Fake Detection and Loss
        pred_fake_pool = self.discriminate(input_label, fake_image, use_pool=True)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)        

        # Real Detection and Loss        
        pred_real = self.discriminate(input_label, real_image)
        loss_D_real = self.criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)        
        pred_fake = self.netD.forward(torch.cat((input_label, fake_image), dim=1))        
        loss_G_GAN = self.criterionGAN(pred_fake, True)               
        
        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                        self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat
                   
        # VGG feature matching loss
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            # Upweight the thing classes so small vehicles carry gradient proportional to
            # their importance rather than to their pixel count. thing_weight 1.0 is the
            # original uniform loss.
            _tw = float(getattr(self.opt, 'thing_weight', 1.0))
            _wmap = None
            if _tw > 1.0:
                _ids = [i for i in MAPILLARY_THING_IDS if i < self.opt.label_nc]
                _things = input_label[:, _ids].sum(dim=1, keepdim=True).clamp(0, 1)
                _wmap = 1.0 + (_tw - 1.0) * _things

            # VEGETATION, WEIGHTED BY DISTANCE.
            # Measured on Town10HD (vegetation-labelled pixels, Laplacian variance):
            #     real training photos   near 1191  far 1246   far/near 1.05
            #     CARLA source           near  928  far 1030   far/near 1.11
            #     v50 render             near  672  far  561   far/near 0.83
            # The model reaches 56% of the photographs' detail, and unlike both references it gets
            # WORSE with distance. The mechanism is that L1 and VGG are area-weighted: a distant
            # tree covers few pixels, contributes almost no gradient, and is never learned. So
            # upweight vegetation, and upweight it MORE the further away it is.
            #
            # The depth channel is inverse depth -- measured sky 0.0, road ~170 of 255 -- so FAR is
            # the LOW end and the distance term is (1 - depth). Getting this backwards would
            # upweight the bonnet instead of the treeline.
            _vw = float(getattr(self.opt, 'veg_weight', 1.0))
            if _vw > 1.0:
                _vids = [i for i in MAPILLARY_VEG_IDS if i < self.opt.label_nc]
                _veg = input_label[:, _vids].sum(dim=1, keepdim=True).clamp(0, 1)
                _fb = float(getattr(self.opt, 'far_boost', 0.0))
                _dist = 1.0
                if _fb > 0 and depth is not None and torch.is_tensor(depth):
                    _d = depth.data.cuda().float()
                    if _d.dim() == 3:
                        _d = _d.unsqueeze(1)
                    _dist = 1.0 + _fb * (1.0 - _d.clamp(0, 1))
                _add = (_vw - 1.0) * _veg * _dist
                _wmap = _add + 1.0 if _wmap is None else _wmap + _add

            # ADDITIVE form (--veg_extra). v63 used the weight map above and it worked on what it
            # aimed at -- far/near detail 0.83 -> 1.00 -- but the map feeds a WEIGHTED MEAN, so
            # emphasis on vegetation is dilution of everything else. Measured cost: road high-pass
            # energy 1.81x the parent and 2.1x a real photograph, i.e. invented grain on the one
            # surface lane geometry is read from, present from epoch 8 onward at every checkpoint.
            # Here the original objective is kept at full strength and a vegetation-only term is
            # ADDED, so nothing is diluted.
            _ve = float(getattr(self.opt, 'veg_extra', 0.0))
            if _ve > 0:
                _vids = [i for i in MAPILLARY_VEG_IDS if i < self.opt.label_nc]
                _veg = input_label[:, _vids].sum(dim=1, keepdim=True).clamp(0, 1)
                _fb = float(getattr(self.opt, 'far_boost', 0.0))
                _dist = 1.0
                if _fb > 0 and depth is not None and torch.is_tensor(depth):
                    _d = depth.data.cuda().float()
                    if _d.dim() == 3:
                        _d = _d.unsqueeze(1)
                    _dist = 1.0 + _fb * (1.0 - _d.clamp(0, 1))
                _plain, _veg_loss = self.criterionVGG.forward_plus(
                    fake_image, real_image, _veg * _dist)
                loss_G_VGG = (_plain + _ve * _veg_loss) * self.opt.lambda_feat
            else:
                loss_G_VGG = self.criterionVGG(fake_image, real_image, _wmap) * self.opt.lambda_feat

        # temporal consistency: fake_t should match previous frame warped into t (non-occluded regions)
        loss_G_Temporal = 0
        if getattr(self.opt, 'temporal', False) and prev is not None and flow is not None:
            import torch.nn.functional as _Fn
            flow_c = flow.data.cuda().float()
            warped_prev = self.warp(prev.float(), flow_c)               # prev warped into current frame
            occ = (torch.abs(warped_prev - real_image).mean(1, keepdim=True) < 0.15).float()  # non-occluded mask
            # TEXTURE GATE. Optical flow is meaningless where there is nothing to track -- sky,
            # blank walls, plain road. The warp there drags in whatever edge happens to be
            # nearby, and |fake - warped_prev| then teaches G to PAINT those edges permanently.
            # That is the faint building outline that appeared in the sky of v54/v58 and survived
            # even with the autoregressive feedback fully disabled, i.e. it was baked into the
            # weights rather than accumulated at inference. The occlusion test above does not
            # catch it: in a smooth region the warped and real frames still agree closely, so occ
            # stays 1 exactly where the flow is least trustworthy.
            # Weight by local gradient energy of the REAL frame, so untextured pixels contribute
            # nothing to the temporal loss.
            _g = real_image.mean(1, keepdim=True)
            _gx = _Fn.pad((_g[:, :, :, 1:] - _g[:, :, :, :-1]).abs(), (0, 1, 0, 0))
            _gy = _Fn.pad((_g[:, :, 1:, :] - _g[:, :, :-1, :]).abs(), (0, 0, 0, 1))
            _tex = _Fn.avg_pool2d(_gx + _gy, 9, stride=1, padding=4)
            occ = occ * (_tex > getattr(self.opt, 'temporal_tex_thresh', 0.02)).float()
            denom = occ.sum().clamp(min=1.0)
            loss_G_Temporal = (torch.abs(fake_image - warped_prev) * occ).sum() / denom * self.opt.lambda_temporal

        # video (temporal) discriminator: judge frame PAIRS (prev, cur). A blurry/flickering fake
        # looks fake AS A TRANSITION -> G must be both sharp AND consistent (fixes temporal-only blur).
        loss_G_VGAN = 0; loss_D_V_real = 0; loss_D_V_fake = 0
        if getattr(self.opt, 'video_disc', False) and getattr(self.opt, 'temporal', False) and prev is not None:
            prev_v = prev.to(real_image.dtype)                     # prev already cuda'd above (temporal path)
            real_pair = torch.cat((prev_v, real_image), dim=1)
            fake_pair = torch.cat((prev_v, fake_image), dim=1)
            loss_D_V_fake = self.criterionGAN(self.netD_V.forward(fake_pair.detach()), False)
            loss_D_V_real = self.criterionGAN(self.netD_V.forward(real_pair), True)
            loss_G_VGAN   = self.criterionGAN(self.netD_V.forward(fake_pair), True) * self.opt.lambda_vgan

        # Only return the fake_B image if necessary to save BW
        return [ self.loss_filter( loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake, loss_G_Temporal, loss_G_VGAN, loss_D_V_real, loss_D_V_fake ), None if not infer else fake_image ]

    def inference(self, label, inst, image=None, weather=None, edge=None, prev=None,
                  mf_labels=None, mf_edges=None, depth=None, normal=None, light=None, chroma=None):
        # Encode Inputs
        image = Variable(image) if image is not None else None
        input_label, inst_map, real_image, _ = self.encode_input(Variable(label), Variable(inst), image, infer=True, weather_id=weather, edge_map=edge, depth_map=depth, normal_map=normal, light_map=light, chroma_map=chroma)

        # Fake Generation
        if self.use_features:
            if self.opt.use_encoded_image:
                # encode the real image to get feature map
                feat_map = self.netE.forward(real_image, inst_map)
            else:
                # sample clusters from precomputed features
                feat_map = self.sample_features(inst_map)
            input_concat = torch.cat((input_label, feat_map), dim=1)
        else:
            input_concat = input_label
        # non-autoregressive temporal: stack neighbour frames (REAL labels, not generated -> no drift/blur)
        if getattr(self.opt, 'multiframe', False) and mf_labels is not None:
            blocks = []
            for k in range(len(mf_labels)):
                ek = mf_edges[k] if (mf_edges is not None and k < len(mf_edges)) else None
                blocks.append(input_label if mf_labels[k] is None else self.encode_frame(Variable(mf_labels[k]), ek))
            # Same zero-padding as the training path in forward(). encode_frame emits label+edge
            # only (66ch here) while the current frame's block also carries depth, normal and
            # chroma (73ch), and netG_input_nc is (per-frame width) * n_frames. Without this the
            # stack is 205 against a 219-wide first conv -- training succeeded and inference then
            # failed with exactly that mismatch, because only forward() had been fixed.
            wide = input_concat.size(1)
            for k, b in enumerate(blocks):
                if b.size(1) < wide:
                    pad = torch.zeros(b.size(0), wide - b.size(1), b.size(2), b.size(3),
                                      device=b.device, dtype=b.dtype)
                    blocks[k] = torch.cat([b, pad], dim=1)
            input_concat = torch.cat(blocks, dim=1)
        # temporal: condition on previous generated frame (autoregressive at inference)
        elif getattr(self.opt, 'temporal', False) and prev is not None:
            input_concat = torch.cat((input_concat, prev.data.cuda().to(input_concat.dtype)), dim=1)

        if torch.__version__.startswith('0.4'):
            with torch.no_grad():
                fake_image = self.netG.forward(input_concat)
        else:
            fake_image = self.netG.forward(input_concat)
        return fake_image

    def sample_features(self, inst): 
        # read precomputed feature clusters 
        cluster_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.cluster_path)        
        features_clustered = np.load(cluster_path, encoding='latin1').item()

        # randomly sample from the feature clusters
        inst_np = inst.cpu().numpy().astype(int)                                      
        feat_map = self.Tensor(inst.size()[0], self.opt.feat_num, inst.size()[2], inst.size()[3])
        for i in np.unique(inst_np):    
            label = i if i < 1000 else i//1000
            if label in features_clustered:
                feat = features_clustered[label]
                cluster_idx = np.random.randint(0, feat.shape[0]) 
                                            
                idx = (inst == int(i)).nonzero()
                for k in range(self.opt.feat_num):                                    
                    feat_map[idx[:,0], idx[:,1] + k, idx[:,2], idx[:,3]] = feat[cluster_idx, k]
        if self.opt.data_type==16:
            feat_map = feat_map.half()
        return feat_map

    def encode_features(self, image, inst):
        image = Variable(image.cuda(), volatile=True)
        feat_num = self.opt.feat_num
        h, w = inst.size()[2], inst.size()[3]
        block_num = 32
        feat_map = self.netE.forward(image, inst.cuda())
        inst_np = inst.cpu().numpy().astype(int)
        feature = {}
        for i in range(self.opt.label_nc):
            feature[i] = np.zeros((0, feat_num+1))
        for i in np.unique(inst_np):
            label = i if i < 1000 else i//1000
            idx = (inst == int(i)).nonzero()
            num = idx.size()[0]
            idx = idx[num//2,:]
            val = np.zeros((1, feat_num+1))                        
            for k in range(feat_num):
                val[0, k] = feat_map[idx[0], idx[1] + k, idx[2], idx[3]].data[0]            
            val[0, feat_num] = float(num) / (h * w // block_num)
            feature[label] = np.append(feature[label], val, axis=0)
        return feature

    def get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,:,:-1] = edge[:,:,:,:-1] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,1:,:] = edge[:,:,1:,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        edge[:,:,:-1,:] = edge[:,:,:-1,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        if self.opt.data_type==16:
            return edge.half()
        else:
            return edge.float()

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        if getattr(self.opt, 'video_disc', False):
            self.save_network(self.netD_V, 'D_V', which_epoch, self.gpu_ids)
        if self.gen_features:
            self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())           
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd        
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

class InferenceModel(Pix2PixHDModel):
    def forward(self, inp):
        label, inst = inp
        return self.inference(label, inst)

        
