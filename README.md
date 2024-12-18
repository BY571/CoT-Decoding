# CoT-Decoding
Minimal PyTorch implementation of the Chain-of-Thought decoding strategy presented in [Chain-of-Thought Reasoning without Prompting](https://arxiv.org/pdf/2402.10200):

Conventional reasoning with Large Language Models often relies on specific prompting techniques like few-shot learning. The authors of the paper explore an innovative decoding strategy that uncovers inherent reasoning paths by exploring top-k alternative token sequences. By altering the traditional greedy decoding process, the method reveals that Chain of Thought reasoning can emerge naturally within model-generated sequences, without explicit human-crafted prompts. The approach challenges existing assumptions about reasoning in LLMs by demonstrating that complex reasoning paths are intrinsically present in model outputs.

<details>
<summary><h2>Environment Setup</h2></summary>

### Create a new conda environment
```
conda create -n cot-decoder python=3.9 -y
```
### Activate the environment
```
conda activate cot-decoder
```
### Install PyTorch (adjust based on your CUDA version if using GPU)
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
### Install Hugging Face Transformers
```
pip install transformers
```
### Additional dependencies
```
pip install numpy
```
### Verify installation
```
python -c "import torch; import transformers; print('Installation successful!')"
```
<details>

# TODO: 
Parallelize the generation of the cot-paths.