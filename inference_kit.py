import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

import pandas as pd

def init_model(model_name: str) -> tuple:
    """
    Initialize and return the tokenizer and model for the given model name.

    Args:
        model_name (str): The name of the model to load.
    Returns:
        tuple: A tuple containing (tokenizer, model).
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto", # Enable model sharding to allow for it to fit on the gpus
        torch_dtype=torch.float16,
    )
    return tokenizer, model

def generate(tokenizer: AutoTokenizer, model: AutoModelForCausalLM, messages: list[dict], **kwargs) -> tuple[str, str]:
    """
    Generates a response from the model given a list of messages.

    Args:
        tokenizer (AutoTokenizer): The tokenizer for the model.
        model (AutoModelForCausalLM): The language model.
        messages (list[dict]): A list of messages in the format [{"role": "system"/"user"/"assistant", "content": "text"}].
    
    Returns:
        tuple[str, str]: The trimmed response text and the full response text.

    Examples:
        >>> messages = [
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": "How are you today?"}
        ]
        >>> response = generate(tokenizer, model, messages)
    """
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
		return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    # if kwargs specifies for generation parameters, use them; else use defaults
    max_new_tokens = kwargs.get("max_new_tokens", 1000)
    temperature = kwargs.get("temperature", 0.8)
    do_sample = kwargs.get("do_sample", True)

    outputs = model.generate(
        **inputs,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.9,
        do_sample=do_sample,
    )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # remove everything before and including the first occurrence of "assistant" which should denote the start of the response
    if "assistant" in output_text:
        trimmed_text = output_text.split("assistant")[-1].strip()
        trimmed_text = trimmed_text.split(f"{messages[-1]['content']}")[-1].strip()
    return trimmed_text, output_text

def sum_seq_log_prob(tokenizer: AutoTokenizer, model: AutoModelForCausalLM, string:str) -> float:
    """
    Calculate the log probability of a given string according to the model.

    Args:
        tokenizer (AutoTokenizer): The tokenizer for the model.
        model (AutoModelForCausalLM): The language model.
        string (str): The input string to evaluate.
    
    Returns:
        float: The mean log probability of the input string.
    """
    with torch.no_grad():
        enc = tokenizer(string, return_tensors="pt")
        logits = model(**enc).logits                                # [1, seq, vocab]
        # shift so each position predicts the next token
        shift_logits = logits[:, :-1, :]                            # drop last step
        shift_labels = enc["input_ids"][:, 1:]                      # drop first token
        log_probs = F.log_softmax(shift_logits, dim=-1)             # [1, seq-1, vocab]
        # gather the log-prob of the actually observed next tokens
        gathered = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)  # [1, seq-1]
        return gathered.sum().item()

def mean_seq_log_prob(tokenizer: AutoTokenizer, model: AutoModelForCausalLM, string:str) -> float:
    """
    Calculate the mean log probability over each token of a given string according to the model.

    Args:
        tokenizer (AutoTokenizer): The tokenizer for the model.
        model (AutoModelForCausalLM): The language model.
        string (str): The input string to evaluate.
    
    Returns:
        float: The mean log probability of the input string.
    """
    with torch.no_grad():
        enc = tokenizer(string, return_tensors="pt")
        logits = model(**enc).logits                                # [1, seq, vocab]
        # shift so each position predicts the next token
        shift_logits = logits[:, :-1, :]                            # drop last step
        shift_labels = enc["input_ids"][:, 1:]                      # drop first token
        log_probs = F.log_softmax(shift_logits, dim=-1)             # [1, seq-1, vocab]
        # gather the log-prob of the actually observed next tokens
        gathered = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)  # [1, seq-1]
        return gathered.mean().item()

def get_k_next_tokens(tokenizer: AutoTokenizer, model: AutoModelForCausalLM, prompt: str, k: int) -> list[str]:
    """
    Get the top k next tokens predicted by the model for a given prompt.

    Args:
        tokenizer (AutoTokenizer): The tokenizer for the model.
        model (AutoModelForCausalLM): The language model.
        prompt (str): The input prompt string.
        k (int): The number of top tokens to return.
    
    Returns:
        list[tuple[str, float]]: A list of tuples containing (token, probability) for the top k predictions.
    """
    with torch.no_grad():
        enc = tokenizer(prompt, return_tensors="pt")
        logits = model(**enc).logits                                # [1, seq, vocab]
        last_token_logits = logits[:, -1, :]                        # [1, vocab]
        probs = F.softmax(last_token_logits, dim=-1)                # [1, vocab]
        top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1)   # [1, k]
        top_k_tokens = [tokenizer.decode([idx.item()]) for idx in top_k_indices[0]]
        top_k_probs = top_k_probs[0].tolist()
        return list(zip(top_k_tokens, top_k_probs))

def branch_prompts(tokenizer: AutoTokenizer, model: AutoModelForCausalLM, meta_prompt: str, k: int, n: int) -> pd.DataFrame:
    """
    Generates branched prompts from a meta prompt by expanding out to length n with branching factor k of it's highest probability next tokens.

    Args:
        tokenizer (AutoTokenizer): The tokenizer for the model.
        model (AutoModelForCausalLM): The language model.
        meta_prompt (str): The initial meta prompt to branch from.
        k (int): The branching factor (number of next tokens to consider at each step).
        n (int): The total length of the branched prompts to generate.
        
    Returns:
        pd.DataFrame: A DataFrame containing the branched prompts and their mean log probabilities.
    """

    df = pd.DataFrame(columns=["prompt", "mean_log_prob"])
    prompts = [meta_prompt]

    for step in range(n):
        new_prompts = []
        new_rows = []

        for prompt in prompts:
            next_tokens = get_k_next_tokens(tokenizer, model, prompt, k)

            for token, _ in next_tokens:
                new_prompt = prompt + token
                mean_log_prob = mean_seq_log_prob(tokenizer, model, new_prompt)
                new_prompts.append(new_prompt)
                new_rows.append({"prompt": new_prompt, "mean_log_prob": mean_log_prob})

        # Add all new rows at once
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

        # Optionally show progress
        print(f"Step {step+1}/{n}: {len(new_prompts)} prompts")
        print(df.sort_values(by="mean_log_prob", ascending=False).head(5), "\n")

        prompts = new_prompts

    # Sort final DataFrame by score
    df = df.sort_values(by="mean_log_prob", ascending=False).reset_index(drop=True)
    return df

def evaluate(tokenizer: AutoTokenizer, model: AutoModelForCausalLM, prompt: str, outputs: list[str]):
    """
    Evaluate the combined prompt and response by calculating their log probabilities.

    Args:
        tokenizer (AutoTokenizer): The tokenizer for the model.
        model (AutoModelForCausalLM): The language model.
        sequence (str): The input prompt string.
        outputs (list[str]): The outputs that should be evaluated.

    Returns:
        float: The mean log probability of the combined prompt and response.
    """
    output_means = []
    for output in outputs:
        string = f"""
        user
        
        {prompt}

        assistant
        
        {output}
        """
        output_means.append(mean_seq_log_prob(tokenizer, model, string))
    return sum(output_means) / len(output_means)

def sequence_embedding_similarity(tokenizer: AutoTokenizer, model: AutoModelForCausalLM, seq1: str, seq2: str) -> float:
    """
    Calculate the cosine similarity between the embeddings of two sequences.

    Args:
        tokenizer (AutoTokenizer): The tokenizer for the model.
        model (AutoModelForCausalLM): The language model.
        seq1 (str): The first input sequence.
        seq2 (str): The second input sequence.

    Returns:
        float: The cosine similarity between the embeddings of the two sequences.
    """
    with torch.no_grad():
        enc1 = tokenizer(seq1, return_tensors="pt")
        enc2 = tokenizer(seq2, return_tensors="pt")

        emb1 = model(**enc1).last_hidden_state.mean(dim=1)  # [1, hidden_size]
        emb2 = model(**enc2).last_hidden_state.mean(dim=1)  # [1, hidden_size]

        cos_sim = F.cosine_similarity(emb1, emb2).item()
        return cos_sim

if __name__ == "__main__":
    # Example usage
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer, model = init_model(MODEL_NAME)

    SYSTEM_PROMPT = "You are a helpful assistant."
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {"role": "user", "content": "How do I take the derivative of a cubic function? Please be detailed."}
    ]

    trimmed_response, response  = generate(tokenizer, model, messages, do_sample=False)
    print("Generated response:\n")
    print(trimmed_response)

    print("\nSum Log probability of response:", sum_seq_log_prob(tokenizer, model, response))
    print("\nMean Log probability of response:", mean_seq_log_prob(tokenizer, model, response))
    print("\nMean Log probability of some nonsense:", mean_seq_log_prob(tokenizer, model, "hjal ksf gs ehrgthlewiowlr tgw4yp945 t8w45typ9obp8r9t5wtrl"))
    print("\nMean Log probability of womp womp:", mean_seq_log_prob(tokenizer, model, "womp wmop"))
    prompt = "The capital of France is"
    k = 5
    next_tokens = get_k_next_tokens(tokenizer, model, prompt, k)
    print(f"\nTop {k} next tokens for prompt '{prompt}': {next_tokens}")

    meta_prompt = "Quantum mechanics is the fundamental physical theory that describes"
    k = 2
    n = 8

    from time import time
    print("\nGenerating branched prompts...")
    start_time = time()
    branched_prompts = branch_prompts(tokenizer, model, meta_prompt, k, n)
    end_time = time()
    print(f"Generated {len(branched_prompts)} branched prompts in {end_time - start_time:.2f} seconds.\n")
    print(branched_prompts.to_string(index=False))