import argparse
from models.UNet_model import UNetGenerator
import os
import torch

# Takes a model saved with torch.save(model) and does torch.save(model.module.state_dict())
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='path to model')
    args = parser.parse_args()
    
    print('lol')
    model = torch.load(args.model)
    print('hi')
    torch.save(model.module.state_dict(), '{}_tmp'.format(args.model))
    print('bye')
    os.remove(args.model)
    os.rename('{}_tmp'.format(args.model), args.model)

if __name__ == '__main__':
    main()
