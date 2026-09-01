import os.path
import json
import torch
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        ### weather map (optional)
        self.weather_map = {}
        for candidate in [
            os.path.join(opt.dataroot, opt.phase + '_weather.json'),
            os.path.join(opt.dataroot, 'weather_map.json'),
        ]:
            if os.path.exists(candidate):
                with open(candidate) as f:
                    self.weather_map = json.load(f)
                print('Loaded weather map: %s (%d entries)' % (candidate, len(self.weather_map)))
                break

        ### input A (label maps)
        dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        ### input B (real images)
        if opt.isTrain or opt.use_encoded_image:
            dir_B = '_B' if self.opt.label_nc == 0 else '_img'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)  
            self.B_paths = sorted(make_dataset(self.dir_B))

        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### facade-edge maps (optional)
        if getattr(opt, 'edge_input', False):
            self.dir_edge = os.path.join(opt.dataroot, opt.phase + '_edge')
            self.edge_paths = sorted(make_dataset(self.dir_edge))

        ### monocular-depth maps (optional)
        if getattr(opt, 'depth_input', False):
            self.dir_depth = os.path.join(opt.dataroot, opt.phase + '_depth')
            self.depth_paths = sorted(make_dataset(self.dir_depth))

        ### light/emissive maps (optional)
        if getattr(opt, 'light_input', False):
            self.dir_light = os.path.join(opt.dataroot, opt.phase + '_light')
            self.light_paths = sorted(make_dataset(self.dir_light))

        ### chroma prior (optional)
        if getattr(opt, 'chroma_input', False):
            self.dir_chroma = os.path.join(opt.dataroot, opt.phase + '_chroma')
            self.chroma_paths = sorted(make_dataset(self.dir_chroma))

        ### surface-normal maps (optional)
        if getattr(opt, 'normal_input', False):
            self.dir_normal = os.path.join(opt.dataroot, opt.phase + '_normal')
            self.normal_paths = sorted(make_dataset(self.dir_normal))

        ### load precomputed instance-wise encoded features
        if opt.load_features:                              
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        self.dataset_size = len(self.A_paths) 
      
    def __getitem__(self, index):        
        ### input A (label maps)
        A_path = self.A_paths[index]              
        A = Image.open(A_path)        
        params = get_params(self.opt, A.size)
        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params)
            A_tensor = transform_A(A.convert('RGB'))
        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            A_tensor = transform_A(A) * 255.0

        B_tensor = inst_tensor = feat_tensor = 0
        ### input B (real images)
        if self.opt.isTrain or self.opt.use_encoded_image:
            B_path = self.B_paths[index]   
            B = Image.open(B_path).convert('RGB')
            transform_B = get_transform(self.opt, params)      
            B_tensor = transform_B(B)

        ### if using instance maps        
        if not self.opt.no_instance:
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = transform_A(inst)

            if self.opt.load_features:
                feat_path = self.feat_paths[index]            
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform_A(feat))                            

        ### weather id
        fname = os.path.basename(A_path)
        weather_id = self.weather_map.get(fname, self.weather_map.get(os.path.splitext(fname)[0], 0))
        weather_tensor = torch.LongTensor([int(weather_id)])

        ### facade-edge map (1ch in [0,1], same spatial transform as label)
        edge_tensor = 0
        if getattr(self.opt, 'edge_input', False):
            edge = Image.open(self.edge_paths[index]).convert('L')
            edge_tensor = transform_A(edge)   # NEAREST, no normalize -> [0,1], 1xHxW

        ### monocular-depth map (1ch in [0,1], BILINEAR so continuous depth isn't quantized)
        depth_tensor = 0
        if getattr(self.opt, 'depth_input', False):
            depth = Image.open(self.depth_paths[index]).convert('L')
            transform_cont = get_transform(self.opt, params, method=Image.BILINEAR, normalize=False)
            depth_tensor = transform_cont(depth)   # 1xHxW in [0,1]

        ### surface-normal map (3ch in [0,1], BILINEAR, same spatial transform as label)
        normal_tensor = 0
        if getattr(self.opt, 'normal_input', False):
            normal = Image.open(self.normal_paths[index]).convert('RGB')
            transform_cont = get_transform(self.opt, params, method=Image.BILINEAR, normalize=False)
            normal_tensor = transform_cont(normal)   # 3xHxW in [0,1]

        ### light map (1ch in [0,1], BILINEAR -- light falls off smoothly, NEAREST would band it)
        light_tensor = 0
        if getattr(self.opt, 'light_input', False):
            light = Image.open(self.light_paths[index]).convert('L')
            transform_cont = get_transform(self.opt, params, method=Image.BILINEAR, normalize=False)
            light_tensor = transform_cont(light)   # 1xHxW in [0,1]

        ### chroma prior (3ch in [0,1], BILINEAR -- it is a smooth colour field, not a mask)
        chroma_tensor = 0
        if getattr(self.opt, 'chroma_input', False):
            chroma = Image.open(self.chroma_paths[index]).convert('RGB')
            transform_cont = get_transform(self.opt, params, method=Image.BILINEAR, normalize=False)
            chroma_tensor = transform_cont(chroma)   # 3xHxW in [0,1]

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor,
                      'feat': feat_tensor, 'path': A_path, 'weather': weather_tensor,
                      'edge': edge_tensor, 'depth': depth_tensor, 'normal': normal_tensor,
                      'light': light_tensor, 'chroma': chroma_tensor}

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'