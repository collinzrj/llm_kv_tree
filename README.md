## LLM KV Tree

LLM KV Tree is a python class that automatically stores kv cache of LLM in a tree format, you don't need to handle the store and loading of kv cache yourself, you can just call `llm_tree_accelerate_next_logit`, and it automatically finds the maximum prefix cached, and stores the kv cache of current tokens

You can also call `llm_tree_accelerate_next_logit_batch` for batch processing, it automatically finds the maximum length `n` such that `tokens[:n]` of all the tokens in the batch are cached

LLM KV Tree is especially useful when you do beam search like decoding/searching in LLM, for which you can reuse the kv cache a lot, and you don't need to handle the storage and loading of kv cache yourself