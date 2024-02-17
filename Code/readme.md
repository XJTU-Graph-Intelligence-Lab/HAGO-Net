# Guidence for running HAGO-Net

To run the model, first you need to create a conda environment. We list the dependencies in the ```env.yml``` file, so you just need to run the following command:
```
conda env create -f env.yaml
```
Then run the command:
```
conda activate mole3.8
```
The model file is stored in the hagonet folder.
+ For QM9 molecular property prediction tasks, we provide the model ```HagoNet_qm9.py```.

To load and preprocess the dataset:
+ For QM9 molecular property prediction tasks, use ```qm9ev.py``` to load the dataset.

For the dynamics simulation task, we will release the latest version code of DHAGO-Net after the publication of our relevant work.
For the R/S classfication task, you can download the environment from https://github.com/keiradams/ChIRo, modify the files and follow the instruction of the guidence of ChIRo.


The dataset files are supposed to be downloaded automatically, but if the download fails then you have to download them manually. We list the website where you can download the QM9 and MD17 dataset files:
QM9: https://github.com/klicperajo/dimenet/raw/master/data/qm9_eV.npz
MD17: http://quantum-machine.org/gdml/data/npz/

To make the code-running process easier, we provode a simple example to run the md17 energy & force task on aspirin. Check the files and run the following command:

```
export CUDA_VISIBLE_DEVICES=0  # specify the GPU device you use
python run_qm9.py --property aspirin  # you can choose the molecule or property to run
```
The result would be saved in the ```results``` folder.

For all the tasks, the detailed configuration is listed in the appendix of our article, *HAGO-Net: Hierarchical Geometric Massage Passing for Molecular Representation Learning*.