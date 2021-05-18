#!/bin/bash 
# rsync -av --progress sourcefolder /destinationfolder --exclude thefoldertoexclude --exclude anotherfoldertoexclude

rsync -av --progress -e ssh --exclude='*.h5' --exclude='*.jpg' --exclude='/DATASETS' --exclude='RETINANET/__pycache__/' --exclude='RETINANET/snapshots*' --exclude='/MODELS' --exclude='TF2/examples' --exclude='RETINANET/keras-retinanet' --exclude='/snapshots*' --exclude='.*' group00@158.109.75.51:/home/group00/working/Antoni_THESIS/ .
