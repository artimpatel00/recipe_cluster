"""
Cluster recipes based off word embeddings from pre-trained model
"""

import json
import pickle
from typing import List
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import glob


def remove_ads(ingredients_list: List[List[str]]) -> List[List[str]]:
    """
    remove ads from ingredients list and output new ingredients list without ads
    currently not used for embeddings
    """
    new_ingredients_list = []
    for recipe in ingredients_list:
        new_recipe = []
        if recipe is not None:
            for ingredient in recipe:
                new_recipe.append(ingredient.replace("ADVERTISEMENT", ""))
            new_ingredients_list.append(new_recipe)
        else:
            new_ingredients_list.append([])
    return new_ingredients_list

class RecipeCluster:
    """
    imports recipes, encodes into embeddings, performs dimension reduction, and clusters
    """

    def __init__(self, config_dict):
        # data config
        self.recipes_raw_path = "./recipes_raw/*.json"
        self.remove_ads_from_ingredients = True
        self.imported_recipes_df = "./data/imported_recipes_df.csv"
        self.embeddings_recipes_df = "./data/embeddings_recipes_df.csv"
        self.pc_recipes_df = "./data/pc_reciped_df.csv"
        self.clustered_recipes_df = "./data/clustered_recipes_df.csv"

        # embeddings config
        self.config_dict = {
            "pre_trained_model_name": "all-MiniLM-L6-v2",
            "embeddings_size": 384,
            "use_saved_embeddings": True,
            "sample_to_encode": None,
            "saved_embeddings_path": "./data/embeddings.pkl",
            "num_clusters": 4,
            "cluster_max_dist": 0.33,
            "cluster_min_samples": 15
        }

        self.config_dict.update(config_dict)

    def import_data(self):
        """
        import data
        """
        # get recipes from path
        recipe_paths = glob.glob(self.recipes_raw_path)

        # make one dictionary from all recipe datasets
        all_recipes = []
        for path in recipe_paths:
            with open(path, 'r') as f:
                recipes = json.load(f)
                recipes = list(recipes.items())
            all_recipes += recipes
        all_recipes = dict(all_recipes)
        
        # make dataframe
        ids = [*all_recipes]
        titles_list = [all_recipes[id].get("title", None) for id in ids]
        ingredients_list = [all_recipes[id].get("ingredients", None) for id in ids]
        instructions_list = [all_recipes[id].get("instructions", None) for id in ids]

        # remove ads from ingredients list
        new_ingredients_list = remove_ads(ingredients_list)

        recipes_df = pd.DataFrame({
            "id": ids, 
            "titles": titles_list, 
            "ingredients": new_ingredients_list, 
            "instructions": instructions_list
            })

        # remove recipes where there are no ingredients
        recipes_df = recipes_df[recipes_df["ingredients"].apply(lambda x: len(x)>0)]

        # save to data folder
        recipes_df.to_csv(self.imported_recipes_df, index=False)

        return recipes_df

    def encode_embeddings(self):
        """
        use sentence transformers to get word embeddings
        optionally if embeddings are already saved, then those can be imported
        """

        if self.config_dict["use_saved_embeddings"]:
            with open(self.config_dict["saved_embeddings_path"], "rb") as f:
                data = pickle.load(f)
            recipes_df = pd.DataFrame({
                "id": data["id"], 
                "titles": data["titles"],
                "ingredients": data["ingredients"],
                "instructions": data["instructions"]
                })
            embeddings_df = pd.DataFrame(data["embeddings"])
            embeddings_df.columns = [str(i) for i in range(0,self.config_dict["embeddings_size"])]
            recipes_df = pd.concat([recipes_df, embeddings_df], axis=1)
        
        else:
            recipes_df = pd.read_csv(self.imported_recipes_df)
            model = SentenceTransformer(self.config_dict["pre_trained_model_name"])
            if self.config_dict["sample_to_encode"] is None:
                self.config_dict["sample_to_encode"] = recipes_df.shape[0]
            embeddings = model.encode(
                list(recipes_df["instructions"])[:self.config_dict["sample_to_encode"]], 
                convert_to_tensor=True
                )
            embeddings_df = pd.DataFrame(embeddings)
            embeddings_df.columns = [str(i) for i in range(0,self.config_dict["embeddings_size"])]
            recipes_df = pd.concat([recipes_df, embeddings_df], axis=1)

        recipes_df.to_csv(self.embeddings_recipes_df, index=False)

        return recipes_df

    def dimension_reduction(self):
        """
        use PCA for dimension reduction to retain combination of components with highest variance
        """

        recipes_df = pd.read_csv(self.embeddings_recipes_df)
        recipes_df = recipes_df.dropna()
        # TO DO update embedding size according to model selected
        # scale before PCA
        embedding_columns = [str(i) for i in range(0,self.config_dict["embeddings_size"])]
        x = recipes_df.loc[:, embedding_columns].values
        x = StandardScaler().fit_transform(x)

        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(x)
        principal_df = pd.DataFrame(
            data = principal_components, 
            columns = ["principal_component_one", "principal_component_two"]
            )
            
        recipes_df = pd.concat([recipes_df, principal_df], axis=1)
        
        recipes_df.to_csv(self.pc_recipes_df, index=False)
        return recipes_df

    def cluster(self):
        """
        Use density based clustering on pca components
        """

        recipes_df = pd.read_csv(self.pc_recipes_df)

        db = DBSCAN(
            eps=self.config_dict["cluster_max_dist"], 
            min_samples=self.config_dict["cluster_min_samples"]).fit(
                recipes_df[["principal_component_one", "principal_component_two"]]
            )
        recipes_df["cluster"] = db.labels_
        
        recipes_df.to_csv(self.clustered_recipes_df, index=False)
        return recipes_df

    def __call__(self, use_saved_embeddings, sample_size, cluster_max_dist, cluster_min_samples):

        self.config_dict["use_saved_embeddings"] = use_saved_embeddings
        self.config_dict["sample_to_encode"] = sample_size
        self.config_dict["cluster_max_dist"] = cluster_max_dist
        self.config_dict["cluster_min_samples"] = cluster_min_samples

        recipes_df = self.import_data()
        recipes_df = self.encode_embeddings()
        recipes_df = self.dimension_reduction()
        recipes_df = self.cluster()

        return recipes_df