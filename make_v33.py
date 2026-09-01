"""Reusable v33 combine: warm(DVP, baseline, 0.3) + k*highfreq(baseline). Writes {prefix}_v33_sunny.mp4
(k0.7) and {prefix}_v33_sharp.mp4 (k1.0), prints flick/sharp metrics.
Usage: python make_v33.py <baseline_mp4> <dvp_output_dir> <out_prefix> <tag>"""
import cv2, glob, sys, numpy as np
baseline_mp4, dvp_dir, prefix, tag = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

def rd(p):
    c=cv2.VideoCapture(p); fs=[]
    while True:
        ok,f=c.read()
        if not ok: break
        fs.append(f.astype(np.float32))
    c.release(); return fs

BASEv=rd(baseline_mp4)
if not BASEv:
    print(f'{tag}-RESULT ERROR: empty baseline {baseline_mp4}'); sys.exit(1)
h,w=BASEv[0].shape[:2]
dd=sorted(glob.glob(f'{dvp_dir}/**/0020/out_main_*.jpg',recursive=True)) or sorted(glob.glob(f'{dvp_dir}/**/out_main_*.jpg',recursive=True))
if not dd:
    print(f'{tag}-RESULT ERROR: no DVP frames in {dvp_dir}'); sys.exit(1)
DVP=[cv2.resize(cv2.imread(f).astype(np.float32),(w,h)) for f in dd]
n=min(len(BASEv),len(DVP)); BASEv,DVP=BASEv[:n],DVP[:n]

def hf(b): return b-cv2.GaussianBlur(b,(0,0),2.0)
def warm(v,b,s):
    o=np.empty_like(v)
    for c in range(3):
        vm,vs=v[...,c].mean(),v[...,c].std()+1e-3; bm,bs=b[...,c].mean(),b[...,c].std()
        o[...,c]=(v[...,c]-vm)*(1+s*(bs/vs-1))+bm
    return o
def metric(fn):
    pr=None;fl=[];sh=[]
    for i in range(n):
        o=np.clip(fn(i),0,255).astype(np.uint8);g=cv2.cvtColor(o,cv2.COLOR_BGR2GRAY)
        sh.append(cv2.Laplacian(g,cv2.CV_64F).var())
        if pr is not None: fl.append(np.abs(g.astype(np.float32)-pr).mean())
        pr=g.astype(np.float32)
    return np.mean(fl),np.mean(sh)

bf,bs=metric(lambda i: BASEv[i]); print(f'{tag}-RESULT baseline: flick={bf:.2f} sharp={bs:.0f}')
for k,suf in [(0.7,'v33_sunny'),(1.0,'v33_sharp')]:
    fn=lambda i,kk=k: warm(DVP[i],BASEv[i],0.3)+kk*hf(BASEv[i])
    fl,sh=metric(fn); print(f'{tag}-RESULT {suf} (k{k}): flick={fl:.2f} sharp={sh:.0f}')
    vw=cv2.VideoWriter(f'{prefix}_{suf}.mp4',cv2.VideoWriter_fourcc(*'mp4v'),30,(w,h))
    for i in range(n): vw.write(np.clip(warm(DVP[i],BASEv[i],0.3)+k*hf(BASEv[i]),0,255).astype(np.uint8))
    vw.release()
