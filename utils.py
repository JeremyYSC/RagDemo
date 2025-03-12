import os
import constants

def __get_path(model_name: str) -> str:
    return os.path.join(constants.MODEL_PATH, model_name)

def get_embedding_path():
    return __get_path(constants.EMBEDDING_NAME)

def get_reranker_path():
    return __get_path(constants.RERANKER_NAME)