import torch
import torch.nn.functional as F
from typing import List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

class CoTDecoder:
    def __init__(self, model_name: str):
        """
        Initialize the CoT decoder with a pre-trained model
        
        :param model_name: Hugging Face model identifier
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()

    def compute_token_uncertainty(
        self, 
        logits: torch.Tensor
    ) -> float:
        """
        Compute uncertainty between top two tokens at a decoding step
        
        :param logits: Model logits for the current decoding step
        :return: Uncertainty value
        """
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Get top 2 tokens and their probabilities
        top_probs = torch.topk(probs, k=2, dim=-1).values
        
        # Compute uncertainty as difference in probabilities
        uncertainty = torch.abs(top_probs[0] - top_probs[1]).item()
        
        return uncertainty

    def cot_generate(
        self, 
        prompt: str, 
        top_k: int = 10, 
        max_length: int = 200
    ) -> List[Tuple[List[int], float]]:
        """
        CoT-Decoding generation method with top-k initial expansion and uncertainty tracking 
        as introduced in the paper: Chain-of-Thought Reasoning without Prompting (https://arxiv.org/pdf/2402.10200) 
        
        :param prompt: Input prompt
        :param top_k: Number of initial paths to explore
        :param max_length: Maximum generation length
        :return: List of (token_sequence, confidence) tuples
        """
        # Tokenize input
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        
        # Initial decoding step to get top-k first tokens
        with torch.no_grad():
            outputs = self.model(input_ids)
            first_step_logits = outputs.logits[0, -1, :]
            
            # Get top-k tokens for first step
            top_k_first_tokens = torch.topk(first_step_logits, k=top_k).indices
        
        # Initialize results list
        generation_results = []
        
        # Explore each top-k initial token path
        for k, first_token in enumerate(top_k_first_tokens):
            # Initialize current sequence and uncertainty tracking
            current_sequence = torch.cat([input_ids, first_token.unsqueeze(0).unsqueeze(0)], dim=1)
            path_uncertainties = []
            
            # Greedy decoding for the rest of the tokens
            for _ in range(max_length - current_sequence.shape[1]):
                with torch.no_grad():
                    outputs = self.model(current_sequence)
                    logits = outputs.logits[0, -1, :]
                    
                    # Compute uncertainty for this step
                    step_uncertainty = self.compute_token_uncertainty(logits)
                    path_uncertainties.append(step_uncertainty)
                    
                    # Select top token (greedy decoding) and append to sequence
                    next_token = torch.argmax(logits).unsqueeze(0).unsqueeze(0)
                    current_sequence = torch.cat([current_sequence, next_token], dim=1)
                    
                    # Stop if end of sequence token
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
            
            # Compute average uncertainty as confidence metric
            avg_uncertainty = sum(path_uncertainties) / len(path_uncertainties) if path_uncertainties else 0
            
            # Store result
            generation_results.append((
                current_sequence[0].tolist(), 
                1 - avg_uncertainty  # Convert uncertainty to confidence
            ))
        
        return generation_results

    def decode_and_print_results(
        self, 
        prompt: str, 
        top_k: int = 5, 
        max_length: int = 50
    ):
        """
        Decode and print results in a human-readable format
        
        :param prompt: Input prompt
        :param top_k: Number of initial paths to explore
        :param max_length: Maximum generation length
        """
        results = self.cot_generate(prompt, top_k, max_length)
        
        for i, (token_sequence, confidence) in enumerate(results, 1):
            decoded_text = self.tokenizer.decode(token_sequence)
            print(f"Path {i}:")
            print(f"Generated Text: {decoded_text}")
            print(f"Confidence: {confidence:.4f}\n")


def main():
    # Choose your desired Hugging Face model
    model_name = "microsoft/Phi-3.5-mini-instruct"
    
    decoder = CoTDecoder(model_name)
    
    print("\n\nExample 1 CoT-Decoding Results:")
    # Example prompt from GSM8K dataset taken from the paper
    prompt = """Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them?"""
    
    # Results of CoT-Decoding strategy
    decoder.decode_and_print_results(prompt, top_k=10, max_length=200)

    # Other example prompts from the paper
    print("\n\nExample 2 CoT-Decoding Results:")
    prompt = """I have 3 apples, my dad has 2 more apples than me, how many apples do we have in total?"""
    decoder.decode_and_print_results(prompt, top_k=10, max_length=75)

    print("\n\nExample 3 CoT-Decoding Results:")
    prompt = """Was Nicolas Cage born in an even or odd year?"""
    decoder.decode_and_print_results(prompt, top_k=10, max_length=50)

if __name__ == "__main__":
    main()