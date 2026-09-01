import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

# test
if not opt.engine and not opt.onnx:
    model = create_model(opt)
    if opt.data_type == 16:
        model.half()
    elif opt.data_type == 8:
        model.type(torch.uint8)
            
    if opt.verbose:
        print(model)
else:
    from run_engine import run_trt_engine, run_onnx
    
prev_gen = None   # temporal: previous generated frame (autoregressive)
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    if opt.data_type == 16:
        data['label'] = data['label'].half()
        data['inst']  = data['inst'].half()
    elif opt.data_type == 8:
        data['label'] = data['label'].uint8()
        data['inst']  = data['inst'].uint8()
    if opt.export_onnx:
        print ("Exporting to ONNX: ", opt.export_onnx)
        assert opt.export_onnx.endswith("onnx"), "Export model file should end with .onnx"
        torch.onnx.export(model, [data['label'], data['inst']],
                          opt.export_onnx, verbose=True)
        exit(0)
    minibatch = 1 
    if opt.engine:
        generated = run_trt_engine(opt.engine, minibatch, [data['label'], data['inst']])
    elif opt.onnx:
        generated = run_onnx(opt.onnx, opt.data_type, minibatch, [data['label'], data['inst']])
    else:        
        weather = data['weather'].cuda() if 'weather' in data else None
        edge = data['edge'].cuda() if ('edge' in data and torch.is_tensor(data['edge'])) else None
        depth = data['depth'].cuda() if ('depth' in data and torch.is_tensor(data['depth'])) else None
        light = data['light'].cuda() if ('light' in data and torch.is_tensor(data['light'])) else None
        chroma = data['chroma'].cuda() if ('chroma' in data and torch.is_tensor(data['chroma'])) else None
        normal = data['normal'].cuda() if ('normal' in data and torch.is_tensor(data['normal'])) else None
        prev = None
        if getattr(opt, 'temporal', False):
            if prev_gen is None:   # first frame: black previous frame
                _, _, H, W = data['label'].size()
                prev_gen = torch.zeros(1, opt.output_nc, H, W)
            prev = prev_gen
            # anti-drift knobs: sharpen &/or scale-down the fed previous frame so autoregressive
            # blur cannot compound. VID2VID_PREV_SCALE=0 -> per-frame (no temporal); 1 -> full autoregressive.
            _scale = float(os.environ.get('VID2VID_PREV_SCALE', '1.0'))
            _sharp = float(os.environ.get('VID2VID_PREV_SHARPEN', '0'))
            if _sharp > 0:
                import torch.nn.functional as _F
                prev = prev + _sharp * (prev - _F.avg_pool2d(prev, 5, 1, 2))   # unsharp mask
            if _scale != 1.0:
                prev = prev * _scale
            prev = prev.clamp(-1, 1)
        mf_labels = mf_edges = None
        if getattr(opt, 'multiframe', False) and 'label_prev' in data:
            # feed REAL neighbour labels (available up front from M2F) -> non-autoregressive, no drift
            mf_labels = [data['label_prev'], None, data['label_next']]
            ep = data['edge_prev'] if torch.is_tensor(data.get('edge_prev', 0)) else None
            en = data['edge_next'] if torch.is_tensor(data.get('edge_next', 0)) else None
            mf_edges = [ep, None, en]
        generated = model.inference(data['label'], data['inst'], data['image'], weather=weather, edge=edge, prev=prev,
                                    mf_labels=mf_labels, mf_edges=mf_edges, depth=depth, normal=normal, light=light, chroma=chroma)
        if getattr(opt, 'temporal', False):
            prev_gen = generated.detach().cpu()   # feed this frame's output into the next
        
    visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                           ('synthesized_image', util.tensor2im(generated.data[0]))])
    img_path = data['path']
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path)

webpage.save()
