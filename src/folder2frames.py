import cv2
# from skimage.transform import resize
import sys
import os
import glob

def main():
    if len(sys.argv) != 3:
        raise Exception('usage: python project1.py <input folder> <output folder>')

    infolder, outfolder = sys.argv[1], sys.argv[2]
    filenames = [filename for filename in glob.glob(os.path.join(infolder, '*.mp4'))]
    # filename = infile[infile.rfind('/') + 1:infile.find('.mp4')]
    for filename in filenames:
        vidcap = cv2.VideoCapture(filename)
        name = filename[filename.rfind('/')+1:filename.find('.mp4')]
        print("converting {} to frames".format(name))
        success, image = vidcap.read()
        count = 0
        while success:
          # image = resize(image, (144, 256), mode='constant', anti_aliasing=True) * 256
          cv2.imwrite('{}/{}-{}.jpg'.format(outfolder, name, count), image)
          success,image = vidcap.read()
          count += 1
          if not success:
            print('Failed reading frame {}'.format(count))

if __name__ == '__main__':
    main()
