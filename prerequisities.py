import os
import sys

def pip_install(*args):
    import subprocess  # nosec - disable B404:import-subprocess check

    cli_args = []
    for arg in args:
        cli_args.extend(str(arg).split(" "))
    subprocess.run([sys.executable, "-m", "pip", "install", *cli_args], check=True)

def main():
    os.environ["GIT_CLONE_PROTECTION_ACTIVE"] = "false"

    pip_install("--pre", "-U", "openvino>=2024.2.0", "--extra-index-url", "https://storage.openvinotoolkit.org/simple/wheels/nightly")
    pip_install("--pre", "-U", "openvino-tokenizers[transformers]", "--extra-index-url", "https://storage.openvinotoolkit.org/simple/wheels/nightly")
    pip_install("--pre", "-U", "openvino_genai", "--extra-index-url", "https://storage.openvinotoolkit.org/simple/wheels/nightly")
    pip_install(
        "-q",
        "--extra-index-url",
        "https://download.pytorch.org/whl/cpu",
        "git+https://github.com/huggingface/optimum-intel.git",
        "git+https://github.com/openvinotoolkit/nncf.git",
        "datasets",
        "accelerate",
        "gradio>=4.19",
        "onnx<1.16.2",
        "einops",
        "transformers_stream_generator",
        "tiktoken",
        "transformers>=4.43.1",
        "faiss-cpu",
        "sentence_transformers",
        "langchain>=0.2.0",
        "langchain-community>=0.2.15",
        "langchainhub",
        "unstructured",
        "scikit-learn",
        "python-docx",
        "pypdf",
        "huggingface-hub>=0.26.5",
    )

if __name__ == '__main__':
    main()