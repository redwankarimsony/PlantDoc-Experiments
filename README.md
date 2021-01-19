# PlantDoc-Experiments
To run the vanilla code, at first clone the #PlantDoc dataset from the [repository](https://github.com/pratikkayal/PlantDoc-Dataset) and change value of `dataset_path:` variable in the `configs/vanilla.yaml` file. Then run the follwing command in linux terminal. Make sure that you have a working installation of PyTorch in your python environment.  
`python "/home/redwan/PlantDoc Leaf Disease Classification/train.py" --config configs/vanilla.yaml`




`{'name': 'plant-doc', 'dataset_path': '/home/redwan/PlantDoc-Dataset', 'device': 'cuda:0', 'batch_size': 16, 'img_size': 512, 'box_size': 512, 'epochs': 20}`

2342 labeled training images found!
237 labeled testing images found!
Epoch 0/19
----------
train Loss: 2.5498 Acc: 0.2810
val Loss: 2.0434 Acc: 0.4093

Epoch 1/19
----------
train Loss: 1.6637 Acc: 0.5047
val Loss: 1.6597 Acc: 0.4515

Epoch 2/19
----------
train Loss: 1.3857 Acc: 0.5687
val Loss: 1.5231 Acc: 0.5105

Epoch 3/19
----------
train Loss: 1.2017 Acc: 0.6217
val Loss: 1.4373 Acc: 0.5232

Epoch 4/19
----------
train Loss: 1.0442 Acc: 0.6772
val Loss: 1.3678 Acc: 0.5570

Epoch 5/19
----------
train Loss: 1.0074 Acc: 0.6712
val Loss: 1.2664 Acc: 0.5992

Epoch 6/19
----------
train Loss: 0.8941 Acc: 0.7135
val Loss: 1.2181 Acc: 0.6076

Epoch 7/19
----------
train Loss: 0.7699 Acc: 0.7481
val Loss: 1.1729 Acc: 0.6329

Epoch 8/19
----------
train Loss: 0.7518 Acc: 0.7703
val Loss: 1.1999 Acc: 0.6160

Epoch 9/19
----------
train Loss: 0.7226 Acc: 0.7733
val Loss: 1.0855 Acc: 0.6624

Epoch 10/19
----------
train Loss: 0.7017 Acc: 0.7929
val Loss: 1.1883 Acc: 0.6160

Epoch 11/19
----------
train Loss: 0.6962 Acc: 0.8006
val Loss: 1.1629 Acc: 0.6203

Epoch 12/19
----------
train Loss: 0.7003 Acc: 0.7758
val Loss: 1.1667 Acc: 0.6160

Epoch 13/19
----------
train Loss: 0.7142 Acc: 0.7835
val Loss: 1.1321 Acc: 0.6582

Epoch 14/19
----------
train Loss: 0.6575 Acc: 0.8083
val Loss: 1.2285 Acc: 0.5907

Epoch 15/19
----------
train Loss: 0.6861 Acc: 0.7827
val Loss: 1.1357 Acc: 0.6245

Epoch 16/19
----------
train Loss: 0.6823 Acc: 0.7912
val Loss: 1.1393 Acc: 0.6245

Epoch 17/19
----------
train Loss: 0.6810 Acc: 0.7886
val Loss: 1.1398 Acc: 0.6203

Epoch 18/19
----------
train Loss: 0.6743 Acc: 0.7972
val Loss: 1.1452 Acc: 0.5992

Epoch 19/19
----------
train Loss: 0.7048 Acc: 0.7788
val Loss: 1.1684 Acc: 0.6076

Training complete in 7m 52s
Best val Acc: 0.662447
