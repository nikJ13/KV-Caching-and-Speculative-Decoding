"""
Speculative Decoding Implementation

Reference Papers:
1. Fast Inference from Transformers via Speculative Decoding (https://arxiv.org/pdf/2211.17192)
2. Accelerating Large Language Model Decoding with Speculative Sampling (https://arxiv.org/pdf/2302.01318)

This implementation follows Algorithm 2 from Paper 2 (DeepMind).
See Theorem 1 for why the rejection sampling preserves the target distribution.
"""
import torch
import transformers
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from termcolor import colored

torch.manual_seed(42)
transformers.set_seed(42)


class SamplingConfig:
    def __init__(self,
                 max_new_tokens: int=50,
                 temperature: float=1.0,
                 lookahead_K: int=3,
                 device: str = "cuda:0",
                 debug: bool = False):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.lookahead_K = lookahead_K
        self.debug = debug
        self.dtype = torch.bfloat16


class SpecDecSamplingConfig(SamplingConfig):
    def __init__(self,
                 target_name: str,
                 draft_name: str):
        super().__init__()
        self.target_name = target_name
        self.draft_name = draft_name


class SpeculativeDecoder:
    def __init__(self, config: SpecDecSamplingConfig):
        """
        Initialize target model, draft model, and tokenizer.
        Set models to eval mode.
        """
        self.config = config
        self.device = config.device
        self.temperature = config.temperature
        
        # TODO: Load models and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.target_name, use_fast=True)
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            config.draft_name,
            torch_dtype=config.dtype,
            device_map=self.device
        ).eval()
        self.target_model = AutoModelForCausalLM.from_pretrained(
            config.target_name,
            torch_dtype=config.dtype,
            device_map=self.device
        ).eval()

    def max_fn(self, x):
        """Max function from paper 2 (f)_+"""
        return torch.clamp(x, min = 0)
    
    def get_distribution(self, logits, temperature, epsilon=1e-8):
        """Get probability distribution from logits"""
        # assuming logits here is of dimension 1 x vocab_size
        # Softmax with temperature
        if temperature <= epsilon:
            # Greedy decoding
            idx = logits.argmax(dim=-1, keepdim=True) # get the token index with the highest prob
            probs = torch.zeros_like(logits) 
            probs.scatter_(dim=-1, index=idx, value = 1) # put 1 in the place where indexes are mentioned in the idx arr
            return probs
        # temperature scaling 
        logits /= temperature
        # normalize
        probs = F.softmax(logits, dim=-1)
        #print("INSIDE GET_DISTRIBUTION",probs.shape)
        return probs
    
    @torch.inference_mode()
    def ar_sample(self, model, tokenized_prompt, max_new_tokens, temperature=1.0):
        """
        Standard autoregressive sampling.
        Returns  generated sequence and temperature temp-normalized probs."""
        # TODO: Implement autoregressive generation
        input_ids = tokenized_prompt.clone()
        generated_tokens = []
        probs_list = []

        for _ in range(max_new_tokens):
            output = model(input_ids=input_ids)
            new_token_logits = output.logits[:,-1,:]
            probs = self.get_distribution(new_token_logits, temperature)
            probs_list.append(probs)
            next_token = torch.multinomial(probs, num_samples=1) # choose the index of a token from the vocab
            generated_tokens.append(next_token)
            input_ids = torch.cat([input_ids, next_token], dim = 1) # concat with the existing input ids (prefix input ids + new token id)

        generated_tokens = torch.cat(generated_tokens, dim=1)

        return generated_tokens, probs_list

    @torch.inference_mode()
    # debug flags are left as is for easier debugging / seeing where outputs diverge
    def sd_sample(self, tokenized_prompt, max_new_tokens, lookahead, temperature):
        """
        Speculative decoding (Algorithm 2 from Paper 2).
        
        Args:
            tokenized_prompt: [batch_size, seq_len]
            max_new_tokens: Total tokens to generate
            lookahead: Number of speculative tokens (K)
            temperature: Sampling temperature
            
        Returns:
            generated_tokens: [batch_size, max_new_tokens]
            acceptance_rate: Fraction of draft tokens accepted
        """
        debug = self.config.debug
        bsz, n = tokenized_prompt.shape
        assert bsz == 1, 'Batch size should be 1'
        target_len = n + max_new_tokens

        current_tokens = tokenized_prompt.clone()

        # Metrics
        accepted_count = 0
        draft_token_num = 0
        n_orig = n

        while n < target_len:
            # HINT: you dont want to overshoot on max_new_tokens
            corrected_lookahead = min(lookahead, target_len - n) # TODO
            # TODO: Generate K draft tokens
            draft_tokens, draft_probs_list = self.ar_sample(self.draft_model, current_tokens, corrected_lookahead, temperature)

            if debug:
                drafted_text = self.tokenizer.decode(draft_tokens[0],
                                                     skip_special_tokens=False)
                print(colored(f"Possible continuations: {drafted_text}", 'blue', 'on_black'))


            # TODO: Run target model on draft sequence to verify
            full_sequence = torch.cat([current_tokens, draft_tokens], dim=1)
            target_outputs = self.target_model(input_ids=full_sequence)
            target_logits = target_outputs.logits

            target_logits_for_verification = target_logits[0, n:n+corrected_lookahead, :]
            target_probs_for_verification = self.get_distribution(target_logits_for_verification, temperature)

            draft_token_num += corrected_lookahead

            # TODO: For each draft token, compute acceptance probability and accept/reject

            for t in range(corrected_lookahead):
                # calculate probability distribution
                draft_probs = draft_probs_list[t] # returns probabilities of all the tokens in the vocab for the t-th token prediction
                target_probs = target_probs_for_verification[t:t+1, :]

                draft_token = draft_tokens[:,t] # take the t-th lookahead token id

                target_val = target_probs[0, draft_token].squeeze() # get the probability for the corresponding draft token from the list of probabilities for each token in the vocab
                draft_val = draft_probs[0, draft_token].squeeze()
                accept_prob = torch.min(torch.tensor(1.0, device=self.device), target_val/draft_val)

                r = torch.rand(1, device=self.device)

                # accept loop
                if r < accept_prob:
                    if debug:
                        accepted_token = self.tokenizer.decode(draft_token)
                        print(f"Accepted token: '{accepted_token}'")
                    accepted_count += 1
                    current_tokens = torch.cat([current_tokens, draft_token.unsqueeze(0)], dim=1)
                    n += 1
                # reject loop
                else:
                    # TODO: Reject and resample from adjusted distribution
                    adjusted_prob = self.max_fn(target_probs[0] - draft_probs[0]) # difference between the probabilities of all the logits in the vocab for the particular t-th token
                    adjusted_prob = adjusted_prob / adjusted_prob.sum()

                    new_token = torch.multinomial(adjusted_prob, num_samples=1)
                    current_tokens = torch.cat([current_tokens, new_token.unsqueeze(0)], dim=1)
                    n += 1
                    if debug:
                        rejected_token = self.tokenizer.decode(draft_token)
                        new_token_text = self.tokenizer.decode(new_token[0])
                        print(colored(f"Rejected: {rejected_token}", 'red', 'on_black'))
                        print(colored(f"Replaced with: {new_token_text}", 'green', 'on_black'))
                    break

            # TODO: Sample bonus token if all accepted
            if t == corrected_lookahead - 1 and r < accept_prob:
                bonus_logits = target_logits[:, -1, :]
                bonus_probs = self.get_distribution(bonus_logits, temperature)
                bonus_token = torch.multinomial(bonus_probs, num_samples=1)
                current_tokens = torch.cat([current_tokens, bonus_token], dim=1)
                n += 1


        generated_tokens = current_tokens[:,n_orig:]
        acceptance_rate = accepted_count/draft_token_num if draft_token_num>0 else 0.0
                
        return generated_tokens, acceptance_rate
