"""
Unit tests for recipe clustering
"""
from cluster_recipes.cluster_recipes import RecipeCluster


def test_import_data():
    """
    Checks df shape is correct after importing
    Checks that given dir is real dir
    """

    config_dict = {}
    rc = RecipeCluster(config_dict)
    recipes_df = rc.import_data()

    assert recipes_df.shape == (122971, 4), "df shape is wrong, check import"

def test_encode_embeddings():
    config_dict = {}
    rc = RecipeCluster(config_dict)
    recipe_df = rc.encode_embeddings()

    # when default all-MiniLM-L6-v2 is used, there should be 384 dim in embeddings
    # 384 + 4 existing columns = 388 columns
    assert recipe_df.shape[1] == 388, "embedding dimensions are wrong, check df"

def test_dimenstion_reduction():
    config_dict = {}
    rc = RecipeCluster(config_dict)
    recipe_df = rc.dimension_reduction()
    assert recipe_df.shape[1] == 390, "pca did not work, check df"

def test_cluster():
    config_dict = {}
    rc = RecipeCluster(config_dict)
    recipe_df = rc.cluster()
    # 391 dimensions
    assert recipe_df.shape[1] == 391, "cluster did not work, check df"