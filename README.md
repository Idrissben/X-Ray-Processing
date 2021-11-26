X-Ray-Image Labelling & Reporting
==============================
*Using NLP and computer vision to detect/diagnose and label problems identified in chest x-ray images*


**`Medium blog series:`** [Using Computer Vision and NLP to Caption X-Rays](https://medium.com/@Alexander.Bricken/project-overview-using-computer-vision-and-nlp-to-caption-x-rays-8aad99b27e61)

## Project Overview

Automation has been a major driving force of increased efficiency, reliability, and speed across multiple industries, ranging from banking to transportation to agriculture. In this project, we investigate the potential of deep learning models to automate the process of medical image reporting, looking specifically at chest X-ray images.

Developing a deep learning model to generate/ support the reporting of findings and impressions from X-ray images would be a highly valuable development since it takes radiologists a significant amount of time to carry out this process for a large number of patients. Depending on the level of correctness achieved by the model, it may also be able to reduce human error, which is especially costly in the medical field.

We utilize the [Chest X-Rays Indiana University](https://www.kaggle.com/raddar/chest-xrays-indiana-university?select=indiana_reports.csv) dataset hosted on Kaggle for this project. Our goal is to measure the similarity between our predicted captions and the actual captions provided by doctor's (as presented in the dataset). To do so, we evaluate our model's performance using the [BLEU metric](https://en.wikipedia.org/wiki/BLEU). 

## Project Breakdown
Our process has been broken down into multiple steps (see `notebooks` directory):
- Cleaning the Indiana X-Ray imaging data ([Medium article](https://medium.com/@koredeakande001/part-1-cleaning-and-pre-processing-x-ray-data-647a038e0c33))
- Exploring ways to increase and engineer features for better results.
- Using machine learning, NLP, computer vision, and other methods to label the chest X-Rays.
- Comparing the labels we generate against the actual label provided by the doctors.

---


Repository Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── image_data     <- Data for average frontal and lateral images
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── notebooks          <- Jupyter notebooks. 
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    │
    └── src                <- Source code for use in this project.

Requirements
------------
To setup the project, run the following:
```
pip install -r requirements.txt
```


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
