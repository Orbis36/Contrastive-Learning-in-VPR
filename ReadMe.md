## An All-in-one Framework for VPR task research

We would like to design a framework which can support the famous dataset include:
- MSLS
- Pittsburgh30K/250K
- NordLand
- Oxford-FD
- SF-XL

for the regular compared model like NetVLAD/Patch-NetVLAD/SeqNet/SeqMatchNet

Compare to previous benchmark or framework, we can support both image to image and sequence to sequence model, the cross-modality model will be supported in the future.

As a specific task for the contrastive learning, we will also provide different pretext task for training, for example, classification or triplet loss training

The full structure will be cuda and faiss accelerated.

We will provide Tensorboard and Wandb for the training visualization



