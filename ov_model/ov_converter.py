from pathlib import Path
import subprocess
import os

def convert_embedding_model():
    model_path = os.path.join('..', "model", "bge-m3")
    ov_model_path = "bge-m3" + "-weight-int4"

    export_command_base = f"optimum-cli export openvino --model {model_path} --task sentence-similarity --weight-format int4"
    # export_command_base = f"optimum-cli export openvino --model {model_path} --task sentence-similarity --weight-format int4 --quant-mode int4_f8e4m3 --dataset auto"
    export_command = export_command_base + " " + str(ov_model_path)
    if not Path(ov_model_path).exists():
        subprocess.run(export_command, shell=True)

def convert_rerank_model():
    model_path = os.path.join('..', "model", "bge-reranker-v2-m3")
    ov_model_path = "bge-reranker-v2-m3" + "-weight-int4"

    export_command_base = f"optimum-cli export openvino --model {model_path} --task text-classification --weight-format int4"
    export_command = export_command_base + " " + str(ov_model_path)
    if not Path(ov_model_path).exists():
        subprocess.run(export_command, shell=True)

def convert_models():
    convert_embedding_model()
    convert_rerank_model()

if __name__ == "__main__":
    convert_models()
