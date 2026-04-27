from transformers import AutoTokenizer

PREFIX = "Price is $"
QUESTION = "How much this stream game cost to the nearest dollar?"


class Stream_Tokenizer():
    
    def count_base_tokens(self, stream_item_data: str, tokenizer: AutoTokenizer):
        """
        Count tokens in the base prompt
        """
        return len(tokenizer.encode(stream_item_data, add_special_tokens=False))
    
    def make_prompts(self, stream_item_data: str, price: float, tokenizer: AutoTokenizer, max_tokens: int, do_round: bool, do_truncate: bool):
        """
        Make prompts and completions
        """
        tokens = tokenizer(stream_item_data, add_special_tokens=False)["input_ids"]

        if do_truncate and len(tokens) > max_tokens:
            summary = tokenizer.decode(
                tokens[:max_tokens],
                skip_special_tokens=True
            ).rstrip()
        else:
            summary = stream_item_data

        prompt = f"{QUESTION}\n\n{summary}\n\n{PREFIX}"
        completion = (
            f"{round(price)}.00"
            if do_round
            else str(price)
        )
        return {
            "prompt": prompt,
            "completion": completion}