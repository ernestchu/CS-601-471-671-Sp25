import argparse
import random
import uuid
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning Script")
    parser.add_argument('--model', type=str, required=True,
                        help="Pretrained model name or path (e.g., 'gpt2' or a local path)")
    parser.add_argument('--dataset', type=str, required=True,
                        help="Dataset name or path (e.g., 'databricks/databricks-dolly-15k')")
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help="Learning rate for the optimizer")
    parser.add_argument('--epochs', type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=1,
                        help="Batch size for training")
    parser.add_argument('--gradient_accumulation', type=int, default=8,
                        help="how many steps you accumulate to form a 'large batch'.")
    parser.add_argument('--save_path', type=str, help="path to save the model checkpoint")
    parser.add_argument('--max_length', type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument('--beta', type=float, default=0.01,
                        help="Beta parameter for DPO loss")

    args = parser.parse_args()

    # Load tokenizer and model
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # Ensure a padding token is defined (for models that don't have one by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.train()

    # Load dataset (using the "train" split)
    dataset = load_dataset(args.dataset)["train"]

    # Define a tokenization function that masks out the loss on the prompt
    def tokenize_fn(example):
        # Tokenize the prompt (instructions) and response separately.
        
        instruction = example["prompt"]
        chosen_response = example["chosen"][-1]["content"]
        rejected_response = example["rejected"][-1]["content"]
        id = str(uuid.uuid4())

        # Use add_special_tokens=False so we can control token concatenation
        instr_tokens = tokenizer(instruction, truncation=True, max_length=args.max_length, add_special_tokens=False)
        chosen_resp_tokens = tokenizer(chosen_response, truncation=True, max_length=args.max_length, add_special_tokens=False)
        rejected_resp_tokens = tokenizer(rejected_response, truncation=True, max_length=args.max_length, add_special_tokens=False)

        # Tokenize a separator (here we use "\n\n")
        sep_tokens = tokenizer("\n\n", add_special_tokens=False)["input_ids"]

        # TODO: Concatenate: [instruction] + [separator] + [response], you would need to do it seperately for chosen and rejected input ids
        chosen_input_ids = instr_tokens["input_ids"] + sep_tokens + chosen_resp_tokens["input_ids"]
        rejected_input_ids = instr_tokens["input_ids"] + sep_tokens + rejected_resp_tokens["input_ids"]

        # Create labels: mask out (with -100) the tokens corresponding to the instruction and separator, again you need to do this for both chosen and rejected
        chosen_labels = [-100] * len(instr_tokens["input_ids"]) + [-100] * len(sep_tokens) + chosen_resp_tokens["input_ids"]
        rejected_labels = [-100] * len(instr_tokens["input_ids"]) + [-100] * len(sep_tokens) + rejected_resp_tokens["input_ids"]

        # Then trunctate the inputs / pad the inputs according to args.max_length
        chosen_input_ids = chosen_input_ids[:args.max_length]
        chosen_input_ids += [tokenizer.pad_token_id] * (args.max_length - len(chosen_input_ids))
        chosen_labels = chosen_labels[:args.max_length]
        chosen_labels += [-100] * (args.max_length - len(chosen_labels))
        rejected_input_ids = rejected_input_ids[:args.max_length]
        rejected_input_ids += [tokenizer.pad_token_id] * (args.max_length - len(rejected_input_ids))
        rejected_labels = rejected_labels[:args.max_length]
        rejected_labels += [-100] * (args.max_length - len(rejected_labels))
    
        # Create attention mask
        chosen_attention_mask = [1 if token_id != tokenizer.pad_token_id else 0 for token_id in chosen_input_ids]
        rejected_attention_mask = [1 if token_id != tokenizer.pad_token_id else 0 for token_id in rejected_input_ids]

        # Your code ends here.

        return {"id": id, "chosen_input_ids": chosen_input_ids, "chosen_attention_mask": chosen_attention_mask, "chosen_labels": chosen_labels,
                "rejected_input_ids": rejected_input_ids, "rejected_attention_mask": rejected_attention_mask, "rejected_labels": rejected_labels}

    tokenized_dataset = dataset.map(tokenize_fn, batched=False)
    tokenized_dataset.set_format(type='torch', 
        columns=['id', 'chosen_input_ids', 'chosen_attention_mask', 'chosen_labels', 'rejected_input_ids', 'rejected_attention_mask', 'rejected_labels'])
    
    samples = [random.randint(0, len(dataset) - 1) for _ in range(3)]
    
    dataloader = DataLoader(tokenized_dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # TODO: Function to compute log probabilities for tokens where labels are not -100
    def compute_token_logprobs(logits, input_ids, labels):
        """Compute log probabilities for tokens where labels are not -100"""
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = (labels != -100)

        labels[labels == -100] = 0
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        batch_logprobs = (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        # Your code ends here.

        return batch_logprobs
    
    id2logprobs = {}
    print("Calculating reference log probabilities...")
    
    # TODO: First, loop over the data using the model to get the chosen and rejected log probability
    model.eval()  # Set to eval mode for reference computation
    with torch.no_grad():
        for index, batch in enumerate(dataloader):
            print(f"Processing batch {index+1}/{len(dataloader)}", end="\r")
            
            chosen_input_ids = batch["chosen_input_ids"].to(device)
            chosen_attention_mask = batch["chosen_attention_mask"].to(device)
            chosen_labels = batch["chosen_labels"].to(device)
            
            rejected_input_ids = batch["rejected_input_ids"].to(device)
            rejected_attention_mask = batch["rejected_attention_mask"].to(device)
            rejected_labels = batch["rejected_labels"].to(device)
            
            # Get the logits for the chosen responses
            chosen_logits = model(chosen_input_ids, attention_mask=chosen_attention_mask).logits
            
            # Get the logits for the rejected responses
            rejected_logits = model(rejected_input_ids, attention_mask=rejected_attention_mask).logits
            
            # Compute the log probabilities
            chosen_log_probs = compute_token_logprobs(chosen_logits, chosen_input_ids, chosen_labels)
            rejected_log_probs = compute_token_logprobs(rejected_logits, rejected_input_ids, rejected_labels)
            
            # Store the log probabilities in a dictionary
            for i in range(len(batch["id"])):
                id = batch["id"][i]
                id2logprobs[id] = {
                    "chosen_log_probs": chosen_log_probs[i].detach(),
                    "rejected_log_probs": rejected_log_probs[i].detach(),
                }
    
    print("\nReference log probabilities computed.")
    model.train()  # Set back to train mode
    
    # TODO: DPO Training loop
    beta = args.beta  # Beta parameter for DPO loss
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        print("\nSample generations:")
        for sample_idx in samples:
            sample = dataset[sample_idx]
            prompt = sample["prompt"]
            print(f"\nPrompt: {prompt}")
            # Tokenize the prompt (without response)
            # TODO: paste the code in section 3.1
            tokenized_prompt = tokenizer(prompt, return_tensors="pt").to(device)
            generated_ids = model.generate(
                tokenized_prompt["input_ids"],
                attention_mask=tokenized_prompt["attention_mask"],
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=50,
            )
            generated_text = tokenizer.decode(generated_ids[0])
            print(f"Response: {generated_text[len(prompt):]}")

        
        total_loss = 0.0
        optimizer.zero_grad()  # Zero gradients at the beginning of epoch
        
        for index, batch in enumerate(dataloader):
            chosen_input_ids = batch["chosen_input_ids"].to(device)
            chosen_attention_mask = batch["chosen_attention_mask"].to(device)
            chosen_labels = batch["chosen_labels"].to(device)
            
            rejected_input_ids = batch["rejected_input_ids"].to(device)
            rejected_attention_mask = batch["rejected_attention_mask"].to(device)
            rejected_labels = batch["rejected_labels"].to(device)
            
            # Get the reference log probabilities
            ref_chosen_log_probs = torch.stack([id2logprobs[id]["chosen_log_probs"] for id in batch["id"]])
            ref_rejected_log_probs = torch.stack([id2logprobs[id]["rejected_log_probs"] for id in batch["id"]])
            
            # Compute the logits for the chosen responses
            chosen_logits = model(chosen_input_ids, attention_mask=chosen_attention_mask).logits
            
            # Compute the logits for the rejected responses
            rejected_logits = model(rejected_input_ids, attention_mask=rejected_attention_mask).logits
            
            # Compute token log probabilities
            # Hint: you can call the helper function compute_token_logprobs
            chosen_log_probs = compute_token_logprobs(chosen_logits, chosen_input_ids, chosen_labels)
            rejected_log_probs = compute_token_logprobs(rejected_logits, rejected_input_ids, rejected_labels)            

            # Compute DPO loss cacluation
            # The DPO loss: -log(σ(β(log_prob_difference(x_w) - log_prob_difference(x_l))))
            # Make sure to divide the loss by number of gradient accumulation steps
            loss = -torch.nn.functional.logsigmoid(beta * (
                (chosen_log_probs - rejected_log_probs) -
                (ref_chosen_log_probs - ref_rejected_log_probs)
            )).mean() / args.gradient_accumulation
            loss.backward()
            total_loss += loss.item()
            
            # Gradient Accumulation
            if (index + 1) % args.gradient_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()
                print(f"Batch {index+1}/{len(dataloader)}, Loss: {total_loss}")
                total_loss = 0.0
        
            # Your code ends here.

        # Handle any remaining gradients at the end of epoch
        if total_loss > 0:
            optimizer.step()
            optimizer.zero_grad()
            
        
        # Optional: Save checkpoint at the end of each epoch
        # checkpoint_path = f"{args.save_path}_epoch{epoch+1}"  
        # model.save_pretrained(checkpoint_path)
        # tokenizer.save_pretrained(checkpoint_path)

    # again sample some examples
    print("\nSample generations after training:")
    for sample_index in samples:
        sample = dataset[sample_index]
        prompt = sample["prompt"]
        print(f"\nPrompt: {prompt}")
        # Paste the generation code here.
        tokenized_prompt = tokenizer(prompt, return_tensors="pt").to(device)
        generated_ids = model.generate(
            tokenized_prompt["input_ids"],
            attention_mask=tokenized_prompt["attention_mask"],
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=50,
        )
        generated_text = tokenizer.decode(generated_ids[0])
        print(f"Response: {generated_text[len(prompt):]}")
    
    # Save the final fine-tuned model and tokenizer
    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)

if __name__ == "__main__":
    main()
