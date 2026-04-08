# bin/bash

cd ../../
 
git apply rec_model_zoo_pytorch/RankMixer_ShareMOE/rankmixer-v2.patch
cp -r rec_model_zoo_pytorch/RankMixer/ScalingLaw-Rank-Generic-recall/ScalingLaw rec_model_zoo_pytorch/RankMixer_ShareMOE
