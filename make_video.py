import argparse
import cv2
from models.UNet_model import UNetGenerator
import moviepy.editor as mpe
import numpy as np
from src.utils import VideoInterpTripletsDataset
from torch.utils.data import DataLoader
import os
import sys
from tqdm import tqdm
import torch
sys.path.append("./models")

def write2video(tensor, vid):
    t = tensor.numpy().transpose((1,2,0))
    npt = np.interp(t, (-1.0,1.0), (0,255.0)).astype(np.uint8)
    cvt = cv2.cvtColor(npt, cv2.COLOR_RGB2BGR)
    vid.write(cvt)

def mix_frames(left, right):
    half = int(left.shape[-1] / 2)
    if not left.equal(right):
        right[:, :, :, :half] = left[:, :, :, :half]
    right[:, :, :, half : half + 2] = 1
    return right

"""
half: Left half is original video, every other frame in right half is generated. Original fps.
full: Left half is original video, every frame in right half is generated. Original fps.
double: Generated frame inserted between every frame of original video. Double fps.
slowmo: Generated frame inserted between every frame of original video. Original fps, double playback time.
"""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', help='one of half, full, double, slowmo', default='half')
    parser.add_argument('frames', help='path to frames')
    parser.add_argument('video', help='path to video')
    parser.add_argument('generator', help='path to generator')
    # parser.add_argument('mix', help='mix output between two frames', default=True)
    args = parser.parse_args()
    assert((args.mode == 'half') or (args.mode == 'double'), 'Only half and double modes supported.')

    vidfile = args.video
    filename = vidfile[vidfile.rfind('/') + 1:vidfile.find('.mp4')]
    cam = cv2.VideoCapture(args.video)
    fps = cam.get(cv2.CAP_PROP_FPS)
    print('Original video fps: {}'.format(fps))
    if args.mode == 'double':
        fps *= 2
    dataset = VideoInterpTripletsDataset(args.frames, read_frames=True)
    dataloader = DataLoader(dataset, batch_size=64, num_workers=4)
    height, width = dataset.getsize()
    gen = UNetGenerator()
    gen.load_state_dict(torch.load(args.generator))
    dtype = torch.FloatTensor
    if torch.cuda.is_available():
        gen = gen.cuda()
        dtype = torch.cuda.FloatTensor
    gen.eval()
    outfile = '{}_{}.mp4'.format(filename, args.mode)
    tmp_outfile = '{}_{}tmp.mp4'.format(filename, args.mode)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter(tmp_outfile, fourcc, fps, (width, height))
    with torch.no_grad():
        for index, sample in tqdm(enumerate(dataloader), total=len(dataloader)):
            if args.mode == 'half':
                sample = {k:v[::2, :, :, :] for k,v in sample.items()}
            left, right, mid = sample['left'], sample['right'], sample['out']
            if args.mode == 'half':
                inframes = (left.type(dtype), right.type(dtype))
            elif args.mode == 'double':
                inframes = (left.type(dtype), mid.type(dtype))
            g = gen(inframes).cpu()
            if args.mode == 'half':
                g = mix_frames(mid, g)
                left = mix_frames(left, left)
            for i in range(g.shape[0]):
                write2video(left[i], out)
                write2video(g[i], out)
    out.release()
    cv2.destroyAllWindows()
    if os.path.exists(outfile): 
        os.remove(outfile)
    vid = mpe.VideoFileClip(tmp_outfile)
    orig_vid = mpe.VideoFileClip(args.video)
    final_vid = vid.set_audio(orig_vid.audio)
    final_vid.write_videofile(outfile)
    os.remove(tmp_outfile)

if __name__ == '__main__':
    main()
