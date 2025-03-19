import os
import constants

def _get_model_path(model_name: str) -> str:
    return os.path.join(constants.MODEL_PATH, model_name)

def get_embedding_path():
    return _get_model_path(constants.EMBEDDING_NAME)

def get_reranker_path():
    return _get_model_path(constants.RERANKER_NAME)

def get_vision_language_path():
    return _get_model_path(constants.VISION_LANGUAGE_NAME)

def device_widget(default="AUTO", exclude=None, added=None, description="Device:"):
    import openvino as ov
    import ipywidgets as widgets

    core = ov.Core()

    supported_devices = core.available_devices + ["AUTO"]
    exclude = exclude or []
    if exclude:
        for ex_device in exclude:
            if ex_device in supported_devices:
                supported_devices.remove(ex_device)

    added = added or []
    if added:
        for add_device in added:
            if add_device not in supported_devices:
                supported_devices.append(add_device)

    device = widgets.Dropdown(
        options=supported_devices,
        value=default,
        description=description,
        disabled=False,
    )
    return device