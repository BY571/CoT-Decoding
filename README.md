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
</details>



```
Path 1:
Generated Text: Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them? First, let's determine the price of each type of glass:

- Full price glass: $5
- Discounted glass (every second one): 60% of $5 = 0.60 * $5 = $3

Kylar wants to buy 16 glasses, which means he will have 8 full price glasses and 8 discounted glasses.

Now, let's calculate the total cost:

- Cost for full price glasses: 8 * $5 = $40
- Cost for discounted glasses: 8 * $3 = $24

Total cost
Confidence: 0.1289

Path 2:
Generated Text: Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them? To solve this problem, we need to calculate the total cost of 16 glasses, taking into account the discount on every second glass.

Here's the step-by-step calculation:

1. The first glass costs $5.
2. Every second glass costs 60% of the price of the first glass, which is $5. So, the price of every second glass is 60/100 * $5 = $3.
3. Since Kylar wants to buy 16 glasses, there will be 8 glasses at the full price of $5 and 8 glasses at the discounted
Confidence: 0.1808

Path 3:
Generated Text: Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them?


### Response

Kylar is buying glasses where every second glass is at a discounted price. The first glass costs $5, and the second glass costs 60% of that price.

First, let's calculate the price of the second glass:
60% of $5 = 0.60 * $5 = $3

Now, for every pair of glasses (one full price and one discounted), Kylar pays:
$5 (full price) + $3 (discounted price) = $8

Kylar wants to buy 16 glasses, which
Confidence: 0.1783

Path 4:
Generated Text: Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them? Let's calculate the cost of the glasses step by step:

1. The first glass costs $5.
2. Every second glass costs 60% of the price of the first glass, which is $5. So, the price of every second glass is 60/100 * $5 = $3.

Now, Kylar wants to buy 16 glasses. To find out how many glasses will cost $5 and how many will cost $3, we divide the total number of glasses by 2:

16 glasses / 2 = 8 glasses at $5 each and 8 glass
Confidence: 0.1803

Path 5:
Generated Text: Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them? Kylar wants to buy 16 glasses, and every second glass costs 60% of the price.

Let's calculate the cost of each pair of glasses:

- The first glass costs $5.
- The second glass costs 60% of $5, which is $5 * 0.60 = $3.

So, for every pair of glasses (one full price and one discounted), Kylar pays $5 + $3 = $8.

Since Kylar wants to buy 16 glasses, he will buy 8 pairs of glasses (because 16 divided
Confidence: 0.1842

Path 6:
Generated Text: Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them? The first glass costs $5. Every second glass costs 60% of $5, which is $5 * 0.60 = $3.

For 16 glasses, there will be 8 glasses at the full price and 8 glasses at the discounted price.

The cost for the 8 full-priced glasses is 8 * $5 = $40.
The cost for the 8 discounted glasses is 8 * $3 = $24.

The total cost for 16 glasses is $40 + $24 = $64.

Kyl
Confidence: 0.1086

Path 7:
Generated Text: Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them? We know that every second glass costs 60% of the price, which is $5. So, the price of every second glass is $5 * 0.6 = $3.

Now, let's calculate the total cost for 16 glasses. There are 8 glasses that cost $5 each and 8 glasses that cost $3 each.

The cost for the $5 glasses is 8 * $5 = $40.
The cost for the $3 glasses is 8 * $3 = $24.

The total cost for all 16 glasses is $40 + $24
Confidence: 0.1590

Path 8:
Generated Text: Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them? 

To solve this problem, we need to calculate the total cost of 16 glasses, taking into account the discount on every second glass.

Here's the step-by-step calculation:

1. The first glass costs $5.
2. Every second glass costs 60% of the price of the first glass, which is $5 * 0.60 = $3.
3. Since Kylar wants to buy 16 glasses, there will be 8 glasses at full price and 8 glasses at the discounted price.
4. The total cost for the full-priced glass
Confidence: 0.1796

Path 9:
Generated Text: Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them? If every second glass costs 60% of the price, then the price of every second glass is $5 * 0.6 = $3.

For 16 glasses, there will be 8 glasses at the full price of $5 and 8 glasses at the discounted price of $3.

The total cost for the full-priced glasses is 8 * $5 = $40.
The total cost for the discounted glasses is 8 * $3 = $24.

Adding both amounts together, Kylar needs to pay $40 + $24 = $64
Confidence: 0.1347

Path 10:
Generated Text: Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them? Each glass costs $5, but every second glass costs 60% of the price, which is $5 * 0.60 = $3.

For the first glass, Kylar pays the full price: $5.
For the second glass, he pays 60% of the price: $3.

This pattern repeats for every pair of glasses. Since Kylar wants to buy 16 glasses, there are 8 pairs.

The total cost for each pair (one full price and one discounted price) is $5 + $3 = $8.

For 8 pairs, the total cost
Confidence: 0.1901
```



# TODO: 
Parallelize the generation of the cot-paths.