import os.path
import re
import numpy as np
import cv2
import torch
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image


class MultiframeDataset(BaseDataset):
    """Non-autoregressive temporal dataset. For frame t, also returns label(+edge) of neighbours
    t-1 and t+1 (same video, clamped at boundaries). The model stacks these as generator input and
    predicts only t. With --mf_align, neighbours are optical-flow-warped onto frame t first (using
    the RGB from {phase}_img) so the stacked context is spatially aligned (fixes v29's misalignment).
    Frames named v{vid}_{frame} (training) or pure-numeric (render)."""

    def _key(self, path):
        b = os.path.splitext(os.path.basename(path))[0]
        if b.isdigit():
            return '', int(b)
        m = re.match(r'(.*)_(\d+)$', b)
        if m:
            return m.group(1), int(m.group(2))
        return b, -1

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.align = getattr(opt, 'mf_align', False)

        dir_A = '_A' if opt.label_nc == 0 else '_label'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.has_B = opt.isTrain or opt.use_encoded_image
        if self.has_B:
            dir_B = '_B' if opt.label_nc == 0 else '_img'
            self.B_paths = sorted(make_dataset(os.path.join(opt.dataroot, opt.phase + dir_B)))
        # RGB for optical flow (needed at train AND test when aligning)
        self.img_paths = None
        if self.align:
            self.img_paths = sorted(make_dataset(os.path.join(opt.dataroot, opt.phase + '_img')))
        if getattr(opt, 'edge_input', False):
            self.edge_paths = sorted(make_dataset(os.path.join(opt.dataroot, opt.phase + '_edge')))
        # Conditioning channels for the CURRENT frame. Only the current frame carries these --
        # neighbours contribute label+edge and are zero-padded to width in the model, so no
        # neighbour depth/normal/chroma has to exist.
        #
        # These are NOT optional. encode_input() concatenates one block per enabled flag, and the
        # generator's first conv was built for that exact width. If a flag is on and the loader
        # returns 0 for it, the concat is silently skipped, the input is narrower than the
        # checkpoint's first conv, and load_network drops it to random init WITHOUT raising --
        # the bug that cost v49 its detail. So answer every flag that is on.
        self.extra_dirs = {}
        for flag, (sub, mode) in (('depth_input', ('depth', 'L')),
                                  ('normal_input', ('normal', 'RGB')),
                                  ('light_input', ('light', 'L')),
                                  ('chroma_input', ('chroma', 'RGB'))):
            if getattr(opt, flag, False):
                d = os.path.join(opt.dataroot, opt.phase + '_' + sub)
                if not os.path.isdir(d):
                    raise RuntimeError(f'--{flag} is set but {d} does not exist')
                self.extra_dirs[sub] = (sorted(make_dataset(d)), mode)

        keys = [self._key(p) for p in self.A_paths]
        n = len(keys)
        self.prev_idx = list(range(n)); self.next_idx = list(range(n))
        for i in range(n):
            v, f = keys[i]
            if i > 0 and keys[i - 1][0] == v and keys[i - 1][1] == f - 1:
                self.prev_idx[i] = i - 1
            if i < n - 1 and keys[i + 1][0] == v and keys[i + 1][1] == f + 1:
                self.next_idx[i] = i + 1
        self.dataset_size = n

    def _gray(self, path, w, h):
        im = cv2.imread(path)
        if im is None:
            return np.zeros((h, w), np.uint8)
        if (im.shape[1], im.shape[0]) != (w, h):
            im = cv2.resize(im, (w, h))
        return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    def _warp(self, arr, flow):
        h, w = arr.shape[:2]
        gx, gy = np.meshgrid(np.arange(w), np.arange(h))
        return cv2.remap(arr, (gx + flow[..., 0]).astype(np.float32), (gy + flow[..., 1]).astype(np.float32),
                         cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)

    def _aligned_neighbour(self, nb_idx, cur_idx, kind):
        """Load neighbour label/edge (np) and flow-warp it into the current frame using RGB flow.
        kind: 'label' -> A_paths, 'edge' -> edge_paths. Returns a PIL image ready for transform."""
        paths = self.A_paths if kind == 'label' else self.edge_paths
        nb = np.array(Image.open(paths[nb_idx]).convert('L'))
        h, w = nb.shape[:2]
        if nb_idx != cur_idx and self.img_paths is not None:
            cg = self._gray(self.img_paths[cur_idx], w, h)
            ng = self._gray(self.img_paths[nb_idx], w, h)
            flow = cv2.calcOpticalFlowFarneback(cg, ng, None, 0.5, 3, 21, 3, 5, 1.2, 0)  # cur->nb: pulls nb into cur
            nb = self._warp(nb, flow)
        return Image.fromarray(nb)

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        A = Image.open(A_path)
        params = get_params(self.opt, A.size)
        transform_L = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        transform_B = get_transform(self.opt, params)

        A_tensor = transform_L(A) * 255.0
        B_tensor = transform_B(Image.open(self.B_paths[index]).convert('RGB')) if self.has_B else 0

        pi, ni = self.prev_idx[index], self.next_idx[index]
        if self.align:
            label_prev = transform_L(self._aligned_neighbour(pi, index, 'label')) * 255.0
            label_next = transform_L(self._aligned_neighbour(ni, index, 'label')) * 255.0
        else:
            label_prev = transform_L(Image.open(self.A_paths[pi])) * 255.0
            label_next = transform_L(Image.open(self.A_paths[ni])) * 255.0

        edge_tensor = edge_prev = edge_next = 0
        if getattr(self.opt, 'edge_input', False):
            edge_tensor = transform_L(Image.open(self.edge_paths[index]).convert('L'))
            if self.align:
                edge_prev = transform_L(self._aligned_neighbour(pi, index, 'edge'))
                edge_next = transform_L(self._aligned_neighbour(ni, index, 'edge'))
            else:
                edge_prev = transform_L(Image.open(self.edge_paths[pi]).convert('L'))
                edge_next = transform_L(Image.open(self.edge_paths[ni]).convert('L'))

        # continuous maps: BILINEAR and normalize=False, matching aligned_dataset.py exactly,
        # because the checkpoint was trained against that convention
        extra = {'depth': 0, 'normal': 0, 'light': 0, 'chroma': 0}
        if self.extra_dirs:
            transform_C = get_transform(self.opt, params, method=Image.BILINEAR, normalize=False)
            for sub, (paths, mode) in self.extra_dirs.items():
                if index < len(paths):
                    extra[sub] = transform_C(Image.open(paths[index]).convert(mode))

        return {'label': A_tensor, 'inst': 0, 'image': B_tensor, 'feat': 0,
                'path': A_path, 'weather': torch.LongTensor([0]), 'edge': edge_tensor,
                'label_prev': label_prev, 'label_next': label_next,
                'edge_prev': edge_prev, 'edge_next': edge_next,
                'depth': extra['depth'], 'normal': extra['normal'],
                'light': extra['light'], 'chroma': extra['chroma']}

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'MultiframeDataset'
