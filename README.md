# Recipe Clustering

Recipe Clustering uses word embeddings from pre-trained models and applies dimension reduction, which leaves components that have the most variability. These components are then clustered based on their density using DBSCAN.
This is not meant for training, so data is not shuffled/split. With a more diverse dataset, it can be extended as a labelling method. 

## To Run
```
cd recipes
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## To Run Unit Tests
```
python -m pytest
```