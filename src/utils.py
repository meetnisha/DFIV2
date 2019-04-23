import collections
import glob
import numpy as np
import os
from scipy import misc
import skvideo.io
import torch
from torch.utils.data import Dataset
import cv2

class VideoInterpTripletsDataset(Dataset):
    def __init__(self, directory, read_frames=False, resize=(512,288)):
        """
        :param directory: directory of videos
        :param read_frames: if False, read mp4's. If True, read frames with name 'videoname-framenum.jpg'
        :param resize: sets resize dimensions if we read from a video only important if we don't read_frames
        """
        self.directory = directory
        self.read_frames = read_frames
        self.resize = resize
        if self.read_frames:
            filenames = [filename for filename in glob.glob(os.path.join(directory,'*.jpg'))]
            frames = collections.defaultdict(int)
            for f in filenames:
                #f = f[f.rfind('\\') + 1:f.find('.jpg')] for personal Laptop 
                f = f[f.rfind('\\') + 1:f.find('.jpg')] #for  school GPU
                # print(f)
                file = f[:f.rfind('-')]
                # print(file)
                num = int(f[f.rfind('-') + 1 :])
                if frames[file] < num:
                    frames[file] = num
            self.filenames = [filename for filename in frames]
            #print('Filenames')
            #print(self.filenames)
            frame = misc.imread('{}/{}-{}.jpg'\
                .format(self.directory, self.filenames[0], 0))
            self.height, self.width, _ = frame.shape
            #print('Frame read, (h,w) is ({},{})'.format(self.height, self.width))
            self.frames = [frames[filename] - 2 for filename in self.filenames]
        else:
            self.filenames = [filename for filename in glob.glob(os.path.join(directory,'*.mp4'))]
            self.frames = [int(skvideo.io.ffprobe(f)['video']['@nb_frames']) - 2 for f in self.filenames]
            self.heights = [int(skvideo.io.ffprobe(f)['video']['@height']) for f in self.filenames]
            self.widths = [int(skvideo.io.ffprobe(f)['video']['@width']) for f in self.filenames]
            assert(sum(self.heights) == self.heights[0] * len(self.heights))
            assert(sum(self.widths) == self.widths[0] * len(self.widths))
            if self.resize is not None:
                self.height,self.width = self.resize
            else:
                self.height = self.heights[0]
                self.width = self.widths[0]
        # TODO(wizeng): Implement crop, tensor, and resize transforms
        self.total = sum(self.frames)

    def __len__(self):
        return self.total

    def getsize(self):
        return self.height, self.width

    def __getitem__(self, index):
        '''
        :param index: idx of the first frame in the video you want
        :return: (inframes, outframes)((1,C,H,W,2),(1,C,H,W))
        '''
        # iterate through video's frame lengths to the correct video
        file = 0
        while self.frames[file] <= index:
            index -= self.frames[file]
            file += 1
        if self.read_frames:
            triplet = [misc.imread('{}/{}-{}.jpg'\
                .format(self.directory, self.filenames[file], i)) for i in range(index, index + 3)]
            triplet = [np.interp(trip,(0,255),(-1.0,1.0)) for trip in triplet]
            f = triplet[0]
        else:
            # reader = skvideo.io.vreader(self.filenames[file], inputdict={'--start_number':str(index), '-frames':'3'})
            # reader = skvideo.io.vreader(self.filenames[file], inputdict={'-vf':'select=gte(n\\,{})'.format(index), '-vframes':'3'})
            cap = cv2.VideoCapture(self.filenames[file])
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            triplet = []
            for i in range(3):
                ret,frame = cap.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.resize is not None:
                    frame = cv2.resize(frame,self.resize)
                triplet.append(frame)
            cap.release()
            # for frame in reader:
            #     triplet.append(frame)
        # print("hit before triplet")
        triplet = [torch.from_numpy(frame.transpose((2, 0, 1))).type('torch.FloatTensor') for frame in triplet] # (C, H, W)
        # print(triplet[0].shape)
        return {'left': triplet[0], 'right': triplet[2], 'out': triplet[1]}

        # totalLen = 0
        # correctFilename = None
        # for filename in self.videoFilenames:
        #    lengthOfVideo = int(skvideo.io.ffprobe(filename)['video']['@nb_frames'])
        #    if totalLen+lengthOfVideo-3 >= index:
        #        correctFilename = filename
        #        print(filename)
        #        break
        #    elif totalLen+lengthOfVideo-3 < index and totalLen + lengthOfVideo > index: #weird case
        #        correctFilename = filename
        #        # index - the amount needed to make sure you have at least 3 frames in triplet
        #        index = index - (totalen+lengthOfVideo-index)
        #        break
        #    elif totalLen+lengthOfVideo-3 < index:
        #        lenVideo += lengthOfVideo
        # print(correctFilename)