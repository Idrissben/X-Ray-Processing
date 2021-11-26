# X-Ray-Image Labelling & Reporting
*Using NLP and computer vision to detect/diagnose and label problems identified in chest x-ray images*


**`Medium blog series:`** [Using Computer Vision and NLP to Caption X-Rays](https://medium.com/@Alexander.Bricken/project-overview-using-computer-vision-and-nlp-to-caption-x-rays-8aad99b27e61)

## Project Overview

In summary, we are measuring the similarity of predicted captions to the actual captions provided by doctors.

This is broken down into multiple steps:
- Clean Indiana X-Ray imaging data.
- Explore ways to increase and engineer features for better results.
- Use machine learning, NLP, computer vision, and other methods to label the chest X-Rays.
- Compare the labels we generate against the actual label provided by the doctors.

For example,
- **Real Caption:** the lungs are hyperinflated with coarse interstitial markings compatible with obstructive pulmonary disease and emphysema periodseq there is chronic pleuralparenchymal scarring within the lung bases periodseq no lobar consolidation is seen periodseq no pleural effusion or pneumothorax periodseq heart size is normal period.
- **Prediction Caption:** typical findings of pulmonary consolidation periodseq no pneumothorax periodseq there is no evidence for effusion.

---


Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    │
    └── src                <- Source code for use in this project.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
