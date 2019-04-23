import cv2
# from skimage.transform import resize
import sys

def main():
    if len(sys.argv) != 3:
        print(sys.argv)
        raise Exception('usage: python video2frames.py <video.mp4> <output folder>')

    infile, outfolder = sys.argv[1], sys.argv[2]
    filename = infile[infile.rfind('/') + 1:infile.find('.mp4')]
    print(infile, outfolder, filename)
    vidcap = cv2.VideoCapture(infile)
    success, image = vidcap.read()
    count = 0
    while success:
      # image = resize(image, (144, 256), mode='constant', anti_aliasing=True) * 256
      cv2.imwrite('{}/{}-{}.jpg'.format(outfolder, filename, count), image)  
      success,image = vidcap.read()
      count += 1
      if not success:
        print('Failed reading frame {}'.format(count))
        continue

if __name__ == '__main__':
    main()
