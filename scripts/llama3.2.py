import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

def calculate_entropy(logits):
    valid_logits = logits[torch.isfinite(logits)]
    if len(valid_logits) == 0:
        return torch.tensor(float('inf'))
    probs = F.softmax(valid_logits, dim=-1)
    log_probs = F.log_softmax(valid_logits, dim=-1)
    return -torch.sum(probs * log_probs)

def get_top_k_tokens(logits, k=10):
    valid_logits = logits[torch.isfinite(logits)]
    top_k_values, top_k_indices = torch.topk(valid_logits, min(k, len(valid_logits)))
    return top_k_values, top_k_indices

def branch_search(model, input_ids, max_new_tokens=5, num_branches=3):
    branch_outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_branches,
        do_sample=True,
        output_scores=True,
        return_dict_in_generate=True
    )
    
    branch_sequences = branch_outputs.sequences
    branch_scores = branch_outputs.scores
    
    best_branch = None
    lowest_entropy = float('inf')
    
    for i in range(num_branches):
        branch_entropy = sum(calculate_entropy(score[i]) for score in branch_scores) / len(branch_scores)
        if branch_entropy < lowest_entropy:
            lowest_entropy = branch_entropy
            best_branch = branch_sequences[i]
    
    return best_branch, lowest_entropy

input_text = """
you are a philosopher you think really well and when you feel you are wrong you go 'Wait...' and think even  better!

Here is a challenging question: What is the nature of reality?
"""
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, return_attention_mask=True)

input_ids = inputs.input_ids.to("cuda")
attention_mask = inputs.attention_mask.to("cuda")
model.to("cuda")

max_entropy_threshold = 2.0  # Adjust based on your needs
entropy_increase_threshold = 0.5  # Threshold for significant entropy increase
wait_token_id = tokenizer.encode("Wait...", add_special_tokens=False)[0]

model.eval()
with torch.no_grad():
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=1000,
        output_scores=True,
        return_dict_in_generate=True,
    )

generated_tokens = outputs.sequences[0]
scores = outputs.scores

final_tokens = input_ids[0].tolist()
prev_entropy = None

for idx, score in enumerate(scores):
    logits = score[0]
    entropy = calculate_entropy(logits)
    print(f"\nToken {idx + 1}:")
    print(f"Entropy: {entropy.item():.4f}")
    
    if prev_entropy is not None and entropy - prev_entropy > entropy_increase_threshold:
        print(f"Significant entropy increase detected. Initiating branch search.")
        branch_input = torch.tensor(final_tokens).unsqueeze(0).to("cuda")
        best_branch, branch_entropy = branch_search(model, branch_input)
        print(f"Branch search completed. Branch entropy: {branch_entropy:.4f}")
        if branch_entropy < entropy:
            print("Using branch result.")
            final_tokens.extend(best_branch[len(final_tokens):].tolist())
            break
        else:
            print("Branch not better. Inserting wait token.")
            final_tokens.append(wait_token_id)
    
    if entropy > max_entropy_threshold:
        print(f"Inserting 'Wait...' token due to high entropy.")
        final_tokens.append(wait_token_id)
    
    final_tokens.append(generated_tokens[idx + len(input_ids[0])].item())
    prev_entropy = entropy

generated_text = tokenizer.decode(final_tokens)
print("\nGenerated Text with Reflection and Branching:")
print(generated_text)