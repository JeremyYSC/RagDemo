import os
import constants

def _get_model_path(model_name: str) -> str:
    return os.path.join(constants.MODEL_PATH, model_name)

def _get_openvino_model_path(model_name: str) -> str:
    return os.path.join(constants.OPENVINO_MODEL_PATH, model_name)

def get_embedding_path():
    return _get_model_path(constants.EMBEDDING_NAME)

def get_openvino_embedding_path():
    return _get_openvino_model_path(constants.OPENVINO_EMBEDDING_NAME)

def get_reranker_path():
    return _get_model_path(constants.RERANKER_NAME)

def get_openvino_reranker_path():
    return _get_openvino_model_path(constants.OPENVINO_RERANKER_NAME)

def get_vision_language_path():
    return _get_model_path(constants.VISION_LANGUAGE_NAME)

def get_device():
    """
    device that llm to run on, control by others
    """
    default = "AUTO"
    # import openvino
    # core = openvino.Core()
    # supported_devices = core.available_devices + [default]
    return default