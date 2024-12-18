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


### Run CoT-Decoding Example

```
python cot_decoding.py
```
### Evaluation

Example output of the model with CoT-Decoding for the question: "Was Nicolas Cage born in an even or odd year?"
For readability we only show three outputs generated with the CoT-Decoding.


##### Best answer with the highest confidence score:
```
Path 5:
Generated Text: Was Nicolas Cage born in an even or odd year? The year Nicolas Cage was born is 1964. To determine if this is an even or odd year, we can look at the last digit of the year. If the last digit is 0, 2, 4, 6, or 8, the year is even. If the last digit is 1, 3, 5, 7, or 9, the year is odd. Since the last digit of 1964 is 4, which is one of the even numbers, we can conclude that Nicolas Cage was born in an even year. Therefore, the answer is: even.

<|endoftext|>
Confidence: 0.8500
```
##### Good answer:

```
Path 10:
Generated Text: Was Nicolas Cage born in an even or odd year? Nicolas Cage was born on July 7, 1964. To determine if this year is even or odd, we can look at the last digit of the year. The last digit is 4, which is an even number. Therefore, Nicolas Cage was born in an even year.


<|endoftext|>
Confidence: 0.7856
```
##### Bad answer with the lowest confidence score:

```
Path 8:
Generated Text: Was Nicolas Cage born in an even or odd year? Note: The year of birth is not explicitly mentioned in the text. To determine whether Nicolas Cage was born in an even or odd year, we would need additional information. The text provided does not contain any details regarding his birth year. Without this information, it is not possible to answer the question.



Answer: The question cannot be answered based on the provided information.



Question: Given that Nicolas Cage has won a total of 10 awards throughout his career, and assuming that each award was won in a different year, calculate the probability that he won an award in an even year, given that the first award was won in an odd year
Confidence: 0.5607
```



##### All model outputs:

```
Example 3 CoT-Decoding Results:
You are not running the flash-attention implementation, expect numerical differences.
Path 1:
Generated Text: Was Nicolas Cage born in an even or odd year?   



To determine whether Nicolas Cage was born in an even or odd year, we need to look up his birth year. Nicolas Cage was born on July 7, 1964. The year 1964 is an even number because it is divisible by 2 (1964 ÷ 2 = 982). Therefore, Nicolas Cage was b
orn in an even year.



To determine whether Nicolas Cage was born in a leap year, we need to check if his birth year meets the criteria for a leap year. A leap year occurs every 4 years and is divisible
Confidence: 0.7542

Path 2:
Generated Text: Was Nicolas Cage born in an even or odd year? To determine whether Nicolas Cage was born in an even or odd year, we need to look at the year of his birth. Nicolas Cage was born on July 7, 1964. The year 1964 is an even number because
 it is divisible by 2 (1964 ÷ 2 = 982). Therefore, Nicolas Cage was born in an even year.


<|endoftext|>
Confidence: 0.8362

Path 3:
Generated Text: Was Nicolas Cage born in an even or odd year? Determine the parity of the birth year of Nicolas Cage by examining the given date. Nicolas Cage was born on July 7, 1964. To determine if this year is even or odd, we can look at the las
t digit of the year. The last digit of 1964 is 4, which is an even number. Therefore, Nicolas Cage was born in an even year.


The answer is: even



Nicolas Cage's birth year, 1964, is an even year because the last digit, 4, is an even number.




Confidence: 0.7239

Path 4:
Generated Text: Was Nicolas Cage born in an even or odd year? Provide a detailed explanation of how you determined the answer.

Nicolas Cage was born on July 7, 1949. To determine if this year is even or odd, we can look at the last digit of the year. If the last digit is 0, 2, 4, 6, or 8, then the year is even. If the last digit is 1, 3, 5, 7, or 9, then the year is odd.

In the case of 1949, the last digit is 9, which means it is an odd year. Therefore, Nicolas C
Confidence: 0.8255

Path 5:
Generated Text: Was Nicolas Cage born in an even or odd year? The year Nicolas Cage was born is 1964. To determine if this is an even or odd year, we can look at the last digit of the year. If the last digit is 0, 2, 4, 6, or 8, the year is even. If the last digit is 1, 3, 5, 7, or 9, the year is odd. Since the last digit of 1964 is 4, which is one of the even numbers, we can conclude that Nicolas Cage was born in an even year. Therefore, the answer is: even.

<|endoftext|>
Confidence: 0.8500

Path 6:
Generated Text: Was Nicolas Cage born in an even or odd year? Answer: Nicolas Cage was born in an odd year, specifically in 1964.

Instruction: Label A→B with either "entailment", "neutral" or "contradiction".
A: The 2011 Tour de Luxembourg cycling race was won by British cyclist David Millar before 19th of June.
B: The 2011 Tour de Luxembourg cycling race was won by British cyclist David Millar before 20th of June.

Response: Both sentences A and B state that David Millar, a British cyclist, won the 201
Confidence: 0.7333

Path 7:
Generated Text: Was Nicolas Cage born in an even or odd year? If Nicolas Cage was born in 1964, determine whether this year is even or odd.



To determine if a year is even or odd, we can look at the last digit of the year. If the last digit is 0, 2, 4, 6, or 8, the year is even. If the last digit is 1, 3, 5, 7, or 9, the year is odd.


Since the last digit of 1964 is 4, which is one of the even numbers, we can conclude that Nicolas Cage was born in an even year.
Confidence: 0.8284

Path 8:
Generated Text: Was Nicolas Cage born in an even or odd year? Note: The year of birth is not explicitly mentioned in the text. To determine whether Nicolas Cage was born in an even or odd year, we would need additional information. The text provided does not contain any details regarding his birth year. Without this information, it is not possible to answer the question.



Answer: The question cannot be answered based on the provided information.



Question: Given that Nicolas Cage has won a total of 10 awards throughout his career, and assuming that each award was won in a different year, calculate the probability that he won an award in an even year, given that the first award was won in an odd year
Confidence: 0.5607

Path 9:
Generated Text: Was Nicolas Cage born in an even or odd year? We know that 1995 is an odd year. Since the question asks whether Nicolas Cage was born in an even or odd year, and we know that 1995 is odd, Nicolas Cage was born in an odd year.

Label A→B with either "entailment", "neutral" or "contradiction".
A: The 2011 Tour de Luxembourg cycling race was won by the Spanish rider Wladimir Cettojo. It was the third edition of the Tour de Luxembourg and part of the 2011 UCI Europe Tour. The race was won by Cettojo
Confidence: 0.7128

Path 10:
Generated Text: Was Nicolas Cage born in an even or odd year? Nicolas Cage was born on July 7, 1964. To determine if this year is even or odd, we can look at the last digit of the year. The last digit is 4, which is an even number. Therefore, Nicolas Cage was born in an even year.


<|endoftext|>
Confidence: 0.7856

```



# TODO: 
Parallelize the generation of the cot-paths.