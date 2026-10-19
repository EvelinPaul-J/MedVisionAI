Optional trained weights

The app works without .pth files using its built-in classical explainability fallback.
If you train the four classifiers, place these files here:
  efficientnet_b3.pth
  resnet50.pth
  densenet121.pth
  mobilenet_v3_large.pth
The inference layer automatically uses any compatible weights that are present.
