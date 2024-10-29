from transformers import AutoModelForCausalLM
import torch

class TreeNode:
    def __init__(self):
        self.children = {}
        self.key_value = None  # Stores key_value for this token (n_layers, 2, num_heads, embed_size_per_head)

class LLMKeyValueTree:
    def __init__(self):
        self.root = TreeNode()

    def load_key_values(self, tokens):
        node = self.root
        key_values_list = []
        for token in tokens:
            if token in node.children:
                node = node.children[token]
                if node.key_value is not None:
                    key_values_list.append(node.key_value)
                else:
                    # If key_value is missing, cannot build further
                    break
            else:
                break

        if key_values_list:
            # Stack collected key_values along the sequence_length dimension
            key_values = torch.stack(key_values_list)  # Shape: (sequence_length, n_layers, 2, num_heads, embed_size_per_head)
            return key_values
        else:
            return None

    def store_key_values(self, tokens, key_values):
        # key_values shape: (sequence_length, n_layers, 2, num_heads, embed_size_per_head)
        node = self.root
        sequence_length = key_values.shape[0]
        assert sequence_length == len(tokens), f"Tokens length and key_values sequence length must match. But sequence_length is {sequence_length}, tokens len is {len(tokens)}"

        for i, token in enumerate(tokens):
            if token not in node.children:
                node.children[token] = TreeNode()
            node = node.children[token]
            node.key_value = key_values[i]  # Shape: (n_layers, 2, num_heads, embed_size_per_head)


# past_key_values: (n_layers, 2, batch_size, num_heads, sequence_length, embed_size_per_head) => batch_kv: (batch_size, sequence_length, n_layers, 2, num_heads, embed_size_per_head) 
def permute_to_batch_format(nlayer_kv):
    batch_size, num_heads, sequence_length, embed_size_per_head = nlayer_kv[0][0].shape
    results = []
    for layer_i in range(len(nlayer_kv)):
        for idx in range(len(nlayer_kv[layer_i])):
            results.append(nlayer_kv[layer_i][idx])
    results = torch.cat(results).reshape(len(nlayer_kv), 2, batch_size, num_heads, sequence_length, embed_size_per_head)
    results = results.permute(2, 4, 0, 1, 3, 5)
    return results

# batch_kv: (batch_size, sequence_length, n_layers, 2, num_heads, embed_size_per_head) => past_key_values: (n_layers, 2, batch_size, num_heads, sequence_length, embed_size_per_head) 
def permute_to_nlayer_format(batch_kv):
    past_key_values = batch_kv.permute((2, 3, 0, 4, 1, 5)) # type: ignore
    n_layers = past_key_values.shape[0]
    results = []
    for i in range(n_layers):
        pair = (past_key_values[i][0], past_key_values[i][1])
        results.append(pair)
    return results

def llm_tree_accelerate_next_logit_batch(batch_tokens, tree: LLMKeyValueTree, causal_llm: AutoModelForCausalLM):
    print(batch_tokens)
    batch_tokens = [torch.tensor(tokens) for tokens in batch_tokens]
    assert all(len(lst) == len(batch_tokens[0]) for lst in batch_tokens), "All tokens should have the same length"
    print(f"Tokens length {len(batch_tokens[0])}")
    min_cache_len = 10000000000
    prefix_batch_kv = []
    for tokens in batch_tokens:
        kv = tree.load_key_values(tokens.tolist())
        if kv is None:
            min_cache_len = 0
            break
        min_cache_len = min(kv.shape[0], min_cache_len)
        prefix_batch_kv.append(kv)
    print("min_cache_len", min_cache_len)
    if min_cache_len > 0:
        prefix_batch_kv = torch.stack([kv[:min_cache_len] for kv in prefix_batch_kv])
        suffix_tokens = torch.stack([tokens[min_cache_len:] for tokens in batch_tokens])
        past_kv = permute_to_nlayer_format(prefix_batch_kv)
        outputs = causal_llm(input_ids=suffix_tokens, past_key_values=past_kv, use_cache=True)
        full_batch_kv = permute_to_batch_format(outputs.past_key_values)
        for tokens, kv in zip(batch_tokens, full_batch_kv):
            tree.store_key_values(tokens.tolist(), kv)
    else:
        outputs = causal_llm(input_ids=torch.stack(batch_tokens), use_cache=True)
        batch_kv = permute_to_batch_format(outputs.past_key_values)
        for tokens, kv in zip(batch_tokens, batch_kv):
            tree.store_key_values(tokens.tolist(), kv)
            kv = tree.load_key_values(tokens.tolist())
    return outputs.logits[:, -1, :]


def llm_tree_accelerate_next_logit(tokens, tree: LLMKeyValueTree, causal_llm: AutoModelForCausalLM):
    return llm_tree_accelerate_next_logit_batch([tokens], tree, causal_llm)[0]
