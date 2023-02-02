# MCBC: Minecraft Biome Classifier
The MCBC is our project submission of the 2022/23 "Praktikum Computer Vision" class at the University of Hamburg. It classifies a taken Minecraft screenshot as belonging to one of the following biome groups:
- Aquatic (oceans and rivers)
- Arid (desert, savanna and mesa)
- Forest (jungles, forests and swamps)
- Plains (plains and stony landscapes)
- Snowy (snowy and frozen biomes)

For data generation and evaluation we created the [MCB Data Generator](https://github.com/officiallahusa/mcbc_datagen) and [MCBC Eval](https://github.com/officiallahusa/mcbc_eval). The former for automatically generating a large and highly diverse dataset of Minecraft screenshots, the latter for live evaluation of screenshots taken while running the game.

# Classification Methods
The repository contains one approach utilizing a **convolutional neural network (CNN)** and three **nearest neighbor classifiers** using the euclidean distance over:
- color averages
- 1D-histograms
- 3D-histograms

# Team
Sven Schreiber (https://github.com/svenschreiber) \
Lasse Huber-Saffer (https://github.com/OfficialLahusa) \
Nico Hädicke (https://github.com/Reshxram) \
Lena Kloock (https://github.com/LenaKloock)
