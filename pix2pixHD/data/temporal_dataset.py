import os.path, glob, numpy as np, torch, cv2
from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image

class TemporalDataset(BaseDataset):
    """Sequence dataset for the temporal model. Frames named v<VID>_<FRAME>; for index i the
    'previous' frame is i-1 iff same video prefix (else i itself, with zero flow).
    Returns label_t, edge_t, image_t (real), prev image (real t-1), flow (t-1->t), plus whichever
    of depth/normal/light/chroma the options ask for.

    The extra conditioning channels are NOT optional decoration: encode_input() concatenates one
    channel per enabled flag, and the generator's first conv was built for that exact width. If a
    flag is on and this loader returns 0 for it, the concat is silently skipped, the input is
    narrower than the checkpoint's first conv, and load_network drops that conv to random init
    without raising -- which is what cost v49 its detail. So every flag that is on must be
    answered here with a real tensor.
    """
    # opt flag -> (subdir suffix, PIL mode). Order is irrelevant here; encode_input fixes the
    # concat order (label, weather, edge, depth, light, chroma, normal).
    EXTRA = {'depth_input':  ('depth',  'L'),
             'normal_input': ('normal', 'RGB'),
             'light_input':  ('light',  'L'),
             'chroma_input': ('chroma', 'RGB'),
             'texture_input': ('texture', 'L')}

    def initialize(self, opt):
        self.opt = opt; self.root = opt.dataroot
        self.dir = os.path.join(opt.dataroot, opt.phase)
        self.A_paths = sorted(glob.glob(f'{self.dir}_label/*.png'))
        self.img_dir  = f'{self.dir}_img'
        self.edge_dir = f'{self.dir}_edge'
        self.flow_dir = f'{self.dir}_flow'
        # resolve only the channels actually requested, and fail loudly if one is missing --
        # a silent miss here is the exact failure mode described in the docstring
        self.extra_dirs = {}
        for flag, (sub, mode) in self.EXTRA.items():
            if getattr(opt, flag, False):
                d = f'{self.dir}_{sub}'
                if not os.path.isdir(d):
                    raise RuntimeError(f'--{flag} is set but {d} does not exist')
                self.extra_dirs[sub] = (d, mode)
        self.dataset_size = len(self.A_paths)

    def _vid(self, path):
        return os.path.basename(path).split('_')[0]   # 'v00'

    def __getitem__(self, i):
        A_path = self.A_paths[i]
        stem = os.path.splitext(os.path.basename(A_path))[0]
        A = Image.open(A_path)
        params = get_params(self.opt, A.size)
        t_lab = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        t_img = get_transform(self.opt, params)
        # continuous conditioning maps: BILINEAR so depth/normal/light/chroma are not quantized,
        # normalize=False so they stay in [0,1] -- matching aligned_dataset.py exactly, because
        # the checkpoint was trained against that convention.
        t_cont = get_transform(self.opt, params, method=Image.BILINEAR, normalize=False)
        label = t_lab(A) * 255.0
        image = t_img(Image.open(f'{self.img_dir}/{stem}.jpg').convert('RGB'))
        edge  = t_lab(Image.open(f'{self.edge_dir}/{stem}.png').convert('L'))   # [0,1]

        out = {'depth': 0, 'normal': 0, 'light': 0, 'chroma': 0, 'texture': 0}
        for sub, (d, mode) in self.extra_dirs.items():
            out[sub] = t_cont(Image.open(f'{d}/{stem}.png').convert(mode))

        # previous frame (same video) + flow t-1 -> t. Resize flow to the transformed image size
        # (and scale the flow vectors) so it stays aligned at any loadSize/crop.
        _, Ht, Wt = image.shape
        has_prev = i > 0 and self._vid(self.A_paths[i-1]) == self._vid(A_path)
        # STUDENT FORCING needs the previous frame's CONDITIONING, not just its image, so the
        # model can generate its own t-1 and train on that instead of the real one. Loaded only
        # when asked: it is an extra decode per sample and every other mode ignores it.
        prev_cond = {'label': 0, 'edge': 0, 'depth': 0, 'normal': 0, 'light': 0, 'chroma': 0, 'texture': 0}
        if has_prev:
            pstem = os.path.splitext(os.path.basename(self.A_paths[i-1]))[0]
            prev = t_img(Image.open(f'{self.img_dir}/{pstem}.jpg').convert('RGB'))
            if getattr(self.opt, 'student_forcing', 0) > 0 or getattr(self.opt, 'self_consistency', False):
                prev_cond['label'] = t_lab(Image.open(self.A_paths[i-1])) * 255.0
                prev_cond['edge'] = t_lab(Image.open(f'{self.edge_dir}/{pstem}.png').convert('L'))
                for sub, (d, mode) in self.extra_dirs.items():
                    prev_cond[sub] = t_cont(Image.open(f'{d}/{pstem}.png').convert(mode))
            flow = np.load(f'{self.flow_dir}/{stem}.npy').astype(np.float32)     # h0xw0x2 (t-1->t)
            h0, w0 = flow.shape[:2]
            if 'crop' in self.opt.resize_or_crop:
                # follow the image transform exactly: scale to loadSize width, same crop, same flip (2026-10-30;
                # the old path resized the whole-frame flow onto the CROP, which misaligns it)
                Wf = self.opt.loadSize; Hf = int(round(Wf * A.size[1] / A.size[0]))
                flow = cv2.resize(flow, (Wf, Hf), interpolation=cv2.INTER_LINEAR)
                flow[..., 0] *= float(Wf) / w0; flow[..., 1] *= float(Hf) / h0
                x, y = params['crop_pos']; fs = self.opt.fineSize
                flow = flow[y:y + fs, x:x + fs]
                if flow.shape[:2] != (Ht, Wt):
                    flow = cv2.resize(flow, (Wt, Ht), interpolation=cv2.INTER_LINEAR)
                if self.opt.isTrain and not self.opt.no_flip and params['flip']:
                    flow = flow[:, ::-1].copy(); flow[..., 0] *= -1
            elif (h0, w0) != (Ht, Wt):
                flow = cv2.resize(flow, (Wt, Ht), interpolation=cv2.INTER_LINEAR)
                flow[..., 0] *= float(Wt) / w0
                flow[..., 1] *= float(Ht) / h0
        else:
            prev = image.clone()
            flow = np.zeros((Ht, Wt, 2), np.float32)
        flow_t = torch.from_numpy(np.ascontiguousarray(flow)).permute(2,0,1)   # 2xHxW
        return {'label': label, 'edge': edge, 'image': image, 'prev': prev,
                'flow': flow_t, 'has_prev': int(has_prev), 'inst': 0, 'feat': 0,
                'depth': out['depth'], 'normal': out['normal'],
                'light': out['light'], 'chroma': out['chroma'], 'texture': out['texture'],
                'prev_texture': prev_cond['texture'],
                'prev_label': prev_cond['label'], 'prev_edge': prev_cond['edge'],
                'prev_depth': prev_cond['depth'], 'prev_normal': prev_cond['normal'],
                'prev_light': prev_cond['light'], 'prev_chroma': prev_cond['chroma'],
                'weather': torch.LongTensor([0]), 'path': A_path}

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize
    def name(self): return 'TemporalDataset'
