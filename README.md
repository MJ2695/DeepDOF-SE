# DeepDOF-SE: affordable deep-learning microscopy platform for slide-free histology
Paper pdf will be included soon...

## EDOF network
### Required packages
The required packages can be found in the deepdof-se.yml file. One can also directly use the yml file to create a virtual environment with all the packages. To do so with Anaconda, see this [tutorial](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)

### EDOF dataset
The training, validation, and testing set used for the EDOF network can be found at https://zenodo.org/record/3922596

#### One network EDOF
Uses a single U-net to deconvolve the coded blurred image. Faster to train than the dual U-net version. To use the code, change the file paths at the beginning of each file. Contains the following files:
* **MuseEDOF_cubic_RGB_sep_step1.py**: trains the U-net with a cubic phase mask
* **MuseEDOF_cubic_RGB_sep_step2.py**: trains both the U-net and the phase mask jointly
* **Network_RGB.py**: U-net
* **recon_RGB.py**: Reconstruct captured coded-blurred image after the network is trained and fine-tuned
* **a_zernike_cubic_150mm.mat**: contains the coefficient of the cubic mask
* **zernike_basis_150mm.mat**: contains the Zernike basis of the mask

#### Dual network EDOF
Uses a U-net for each fluorescence dye channel. Higher reconstruction quality. To use the code, change the file paths at the beginning of each file. Contains the following files:
* **CM_MicroDualEDOF_cubic_rms_dualimage_dualunet_128x21_2step_step1.py**: trains dual U-net with a cubic phase mask
* **CM_MicroDualEDOF_cubic_rms_dualimage_dualunet_128x21_2step_step2.py**: trains both the U-net and the phase mask jointly
* **dualunet_reconstruct.py**: Reconstruct captured coded-blurred image after the network is trained and fine-tuned
* **Network_c1.py**: 1 of the 2 U-net
* **Network_c2.py**: 1 of the 2 U-net
* **a_zernike_cubic_150mm.mat**: contains the coefficient of the cubic mask
* **zernike_basis_150mm.mat**: contains the Zernike basis of the mask

## Virtual staining network
The [CycleGAN](https://junyanz.github.io/CycleGAN/) virtual staining network is based on the Tensorflow implementation by Harry Yang [link](https://github.com/leehomyc/cyclegan-1). We will post the code we use soon.

## Virtual staining dataset
This data set contains patient data and is available upon reasonable request. Please contact the corresponding authors Ashok Veeraraghavan or Rebecca Richards-Kortum. 
