#!/usr/bin/env python3
"""
Build a v50 starting checkpoint that grafts the chroma input onto v45 WITHOUT losing v45.

pix2pixHD's load_network (models/base_model.py:71-88) copies a pretrained tensor only when the
shape matches exactly. Widening the input from 70 to 73 channels changes the first conv's shape,
so the whole tensor is dropped and left at random init -- v45's entire learned semantic encoding
along with it. That is what happened to v49: 343,392 params discarded, only 3 of 73 channels
genuinely new, then 2 epochs at lr 5e-5 to relearn it all. Hence its 44% detail loss.

The fix is a channel-wise graft:
    new[:, :70] = v45 weights        <- v45 preserved exactly
    new[:, 70:] = 0                  <- chroma starts contributing nothing

With zero weights the model's output at step 0 is bit-identical to v45, so training can only add.
Gradient still flows into the zero columns (d/dw = input * upstream_grad, which is nonzero), so
chroma is learned rather than frozen out -- the same trick ControlNet uses for its zero convs.

  usage: make_v50_init.py SRC_CKPT DST_CKPT [--new-channels 3] [--g-only]

--g-only grafts the generator alone. Needed for the --temporal graft: pix2pixHD widens ONLY
netG by output_nc there (the previous frame is concatenated for G, not for D), so widening D
too would create the very shape mismatch this script exists to prevent.
"""
import os, re, sys, torch

GRAFT_HINT = 'first-conv weight, widened input'


def graft(src_path, dst_path, new_ch):
    sd = torch.load(src_path, map_location='cpu')
    changed = []
    for k, v in sd.items():
        # first convs are the only tensors whose INPUT channel count depends on the
        # conditioning stack; everything downstream is untouched by adding a channel
        if not k.endswith('.weight') or v.dim() != 4:
            continue
        if k not in ('model.1.weight', 'model1_1.1.weight'):
            continue
        out_c, in_c, kh, kw = v.shape
        w = torch.zeros(out_c, in_c + new_ch, kh, kw, dtype=v.dtype)
        w[:, :in_c] = v                      # v45 preserved, channel for channel
        # w[:, in_c:] stays zero -- chroma contributes nothing at step 0
        sd[k] = w
        changed.append(f'{k}: {tuple(v.shape)} -> {tuple(w.shape)}  '
                       f'({in_c} grafted, {new_ch} zero-init)')
    if not changed:
        raise SystemExit(f'no first-conv tensors found in {src_path} -- refusing to write')
    torch.save(sd, dst_path)
    return changed


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    new_ch = 3
    if '--new-channels' in sys.argv:
        new_ch = int(sys.argv[sys.argv.index('--new-channels') + 1])
        args = [a for a in args if a != str(new_ch)]
    src, dst = args[0], args[1]
    g_only = '--g-only' in sys.argv
    os.makedirs(dst, exist_ok=True)

    for net in (('G',) if g_only else ('G', 'D')):
        s = os.path.join(src, f'latest_net_{net}.pth')
        if not os.path.isfile(s):
            print(f'  {net}: no latest_net_{net}.pth in src, skipped')
            continue
        d = os.path.join(dst, f'latest_net_{net}.pth')
        if net == 'G':
            for line in graft(s, d, new_ch):
                print(f'  G  {line}')
        else:
            # D sees (input stack + image); its first conv widens too, but D is cheap to
            # relearn and a zero graft there is harmless -- do it for symmetry where shapes allow
            sd = torch.load(s, map_location='cpu')
            # the multiscale D names its first conv scale{N}_layer0.0.weight; that is the only
            # tensor whose input width tracks the conditioning stack
            first = re.compile(r'^scale\d+_layer0\.0\.weight$')
            n = 0
            for k, v in list(sd.items()):
                if first.match(k) and v.dim() == 4:
                    out_c, in_c, kh, kw = v.shape
                    w = torch.zeros(out_c, in_c + new_ch, kh, kw, dtype=v.dtype)
                    w[:, :in_c] = v
                    sd[k] = w; n += 1
                    print(f'  D  {k}: {tuple(v.shape)} -> {tuple(w.shape)}')
            if not n:
                raise SystemExit('no D first-conv matched -- refusing to write a silent no-op')
            torch.save(sd, d)
    print(f'\nwrote {dst}')
