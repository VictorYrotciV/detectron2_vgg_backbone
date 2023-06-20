# modified from detectron2 conversion script
import pickle as pkl
import sys
import torch

'''
Usage:
  # download  vgg16 models without BN from torchvision, I only check vgg16 without BN for my research
  wget $URL_IN_TROCHVISION -O vgg16.pth
  # run the conversion
  ./convert-torchvision-to-d2.py vgg16.pth vgg16.pkl


here is vgg16 weights

vgg_block1
features.0.weight
features.0.bias
features.2.weight
features.2.bias

vgg_block2
features.5.weight
features.5.bias
features.7.weight
features.7.bias

vgg_block3
features.10.weight
features.10.bias
features.12.weight
features.12.bias
features.14.weight
features.14.bias

vgg_block4
features.17.weight
features.17.bias
features.19.weight
features.19.bias
features.21.weight
features.21.bias

vgg_block5
features.24.weight
features.24.bias
features.26.weight
features.26.bias
features.28.weight
features.28.bias

classifier[:-1]
classifier.0.weight
classifier.0.bias
classifier.3.weight
classifier.3.bias

do not care
classifier.6.weight
classifier.6.bias

to

backbone.vgg_block1.0.conv1.{bias, weight} 
backbone.vgg_block1.0.conv2.{bias, weight} 
backbone.vgg_block2.0.conv1.{bias, weight} 
backbone.vgg_block2.0.conv2.{bias, weight} 
backbone.vgg_block3.0.conv1.{bias, weight} 
backbone.vgg_block3.0.conv2.{bias, weight} 
backbone.vgg_block3.0.conv3.{bias, weight} 
backbone.vgg_block4.0.conv1.{bias, weight} 
backbone.vgg_block4.0.conv2.{bias, weight} 
backbone.vgg_block4.0.conv3.{bias, weight} 
backbone.vgg_block5.0.conv1.{bias, weight} 
backbone.vgg_block5.0.conv2.{bias, weight} 
backbone.vgg_block5.0.conv3.{bias, weight} 

roi_heads.box_head.fc1.{bias, weight}
roi_heads.box_head.fc2.{bias, weight}

'''


if __name__ == "__main__":
    input = sys.argv[1]

    obj = torch.load(input, map_location="cpu")
    old_v = 0
    vgg_block_id = 1
    conv_ind = 1
    newmodel = {}
    for k in list(obj.keys()):
        k = k.replace("online_net.0.", "")
        parse = k.split('.')
        if parse[0] == 'classifier':
            if parse[1] == '6':
                continue
            fc_id = 1 if parse[1]=='0' else 2
            new_k = f'roi_heads.box_head.fc{fc_id}.{parse[-1]}'
            print(k, "->", new_k)
            newmodel[new_k] = obj.pop(k).detach().numpy()
            continue
        v = int(parse[1])
        if v - old_v > 2:
            vgg_block_id += 1
            conv_ind = 1
        new_k = f'vgg_block{vgg_block_id}.0.conv{conv_ind}.{parse[-1]}' 
        if parse[-1] == 'bias':
            conv_ind += 1
        old_v = v
        print(k, "->", new_k)
        newmodel[new_k] = obj.pop(k).detach().numpy()
    res = {"model": newmodel, "__author__": "torchvision", "matching_heuristics": True} 
    with open(sys.argv[2], "wb") as f:
        pkl.dump(res, f)
    if obj:
        print("Unconverted keys:", obj.keys())

