# Submission Master Thesis "Adversarial Attacks and Robustness in Water Distribution Systems"

This repository contains code, data and results from my Master Thesis. The
contents of the folders are as follows:

- `Data`: networks used for experiments and results from the threshold validation of the leakage detector. Networks in this folder have been specifically adapted (e.g. their duration has been changed).
- `Figures`: All figures that also appeared in the submission
- `Formalizations`: Different formlizations of the least sensitive point problem whidh were initially developed before one was picked.
- `Leakage Detector Plots`: plots from the leakage detector comparison including some plots which were not shown in the thesis
- `Max Residual Change` and `Mean Residual Change`: results from additional experiments concerning the change of detector residuals based on the leak area. These are not directly linked to the results of the algorithms, so they do not appear in the thesis.
- `Network Originals`: the Net1 and Hanoi network before adaptation 
- `Results`: results achieved by the algorithms on the one-week and two-week dataset. These were discussed in the thesis.
- `Taxonomy Trees`: taxonomies for adversarials in Water Distribution Systems and for robustness measures against them.
- `src`: source code

To run the code, I recommend that you set up a `conda` environment using the
same packages as given in `water.yml`. In the best case, this works by running

```
conda env create -f water.yml
```

However, I cannot give a guarantee as this depends on system preferences and
`conda` peculiarities. If the command above does not work, please install at
least the following packages manually into a new environment:

- `wntr`
- `pygad`
- `numpy`
- `scipy`
- `pandas`
- `matplotlib`

If everything is set up correctly, you should be able to navigate to the `src`
folder and run

```
python find_lsp.py
```

to run the Enhanced Genetic Algorithm on the two-week dataset.
All classes and class methods in the `src`-folder contain explanatory
doc-strings. In case of questions, feel free to open an issue.
