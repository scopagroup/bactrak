# BacTrak

BacTrak is a `python` implementation of the tracking algorithm described in the following manuscript:

S. Sarmadi et al. *Stochastic Neural Networks for Automatic Cell Tracking in Microscopy Image Sequences of Bacterial Colonies*. Mathematical and Computational Applications, 27(2):22, 2022 [[arxiv](https://arxiv.org/abs/2104.13482), [doi](https://doi.org/10.3390/mca27020022)].

If you use this code we would appreciate if you add the above reference to your citations.
 

## Examples and Benchmarks

We have included some benchmark files into this repository: Two binary images named 'J.png' (first frame) and 'J_plus.png' (second frame) in the [Sample](Sample) directory. These images are shown below. The software used to generate these synthetic images can be found in the following repository: [https://github.com/jwinkle/eQ](https://github.com/jwinkle/eQ). We have included two synthetic data sets for testing our code:

<p align="center">
<img src="Images/J_labeled_cells.png" alt="J"  width="300"/>
<img src="Images/J_plus_labeled_cells.png" alt="J"  width="300"/>
</p>

To execute the code simply go to the toplevel directory and run the `python` script:
```
python3 BacTrak.py
```

After executing the code the following information will be displayed in your command window: 
``` 
Linking mother and children started

 11%|█         | 551/5000 [00:26<03:35, 20.64it/s]
 
No split registration started

 20%|█▉        | 394/2000 [27:41<1:52:53,  4.22s/it]
``` 


At the end the code will return a python dictionary which is the result of the registration of cells. The dictionary will be saved as `Registration.npy` in the folder [Sample](Sample). The keys of the dictionary are the labels of the cells in 'J' and the values of the dictionary are the labels of the associated cells in 'J_plus'. If a cell devides we have a tuple for the value in the dictionary.

Besides that, we also store pictures to illustrate the registration results inside the [Sample](Sample) folder. The colors of the cells designate the identified mapping. The colors establish a cell-to-cell correspondence between the cells in the image 'J' and the image 'J_plus'. We include an illustration below. 


<p align="center">
<img src="Images/Colored_J.png" alt="J"  width="300"/>
<img src="Images/Colored_J_plus.png" alt="J"  width="300"/>
</p>
 
 

## Dependencies

Our software depends on the following `python` packages:

* `OpenCV` (installation: `pip install opencv-contrib-python`; [https://pypi.org/project/opencv-python](https://pypi.org/project/opencv-python))
* `scikit-image` (installation: `pip install scikit-image`; [https://scikit-image.org](https://scikit-image.org))
* `statsmodels` (installation: `pip install statsmodels`; [https://www.statsmodels.org/stable/index.html](https://www.statsmodels.org/stable/index.html))
* `tqdm` (installation: `pip install tqdm`; [https://pypi.org/project/tqdm](https://pypi.org/project/tqdm))



## References

A BibTeX entry for LaTeX users is
```TeX
@article{Sarmadi:2022a,
author = "S. Sarmadi and J. J. Winkle and R. N. Alnahhas and M. R. Bennett and K. Josic and A. Mang and R. Azencott"
title = "Stochastic Neural Networks for Automatic Cell Tracking in Microscopy Image Sequences of Bacterial Colonies",
journal = "Mathematical and Computational Applications",
volume = "27",
number = "2",
pages = "22",
year = "2022"}
```
