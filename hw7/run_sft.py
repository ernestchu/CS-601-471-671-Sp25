import argparse
import random
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning Script")
    parser.add_argument('--model', type=str, required=True, 
                        help="Pretrained model name or path.")
    parser.add_argument('--dataset', type=str, required=True,
                        help="Dataset name or path.")
    parser.add_argument('--learning_rate', type=float, default=1e-6,
                        help="Learning rate for the optimizer")
    parser.add_argument('--epochs', type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=1,
                        help="Batch size for training")
    parser.add_argument('--gradient_accumulation', type=int, default=9,
                        help="how many steps you accumulate to form a 'large batch'.")
    parser.add_argument('--save_path', type=str, help="path to save the model checkpoint")
    parser.add_argument('--max_length', type=int, default=512,
                        help="Maximum sequence length")

    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.train()
    dataset = load_dataset(args.dataset)["train"]
    dataset = dataset.shuffle(seed=42).select(range(4000))

    # Define a tokenization function that masks out the loss on the prompt
    def tokenize_fn(example):

        # Tokenize the prompt (instructions) and response separately.
        instruction = example["instruction"]
        response = example["response"]

        # Use add_special_tokens=False so we can control token concatenation
        instr_tokens = tokenizer(instruction, truncation=True, max_length=args.max_length, add_special_tokens=False)
        resp_tokens = tokenizer(response, truncation=True, max_length=args.max_length, add_special_tokens=False)

        # Tokenize a separator (here we use "\n\n")
        sep_tokens = tokenizer("\n\n", add_special_tokens=False)["input_ids"]

        # TODO: 
        # First Concatenate: [instruction] + [separator] + [response]
        # Then Create labels: mask out (with -100) the tokens corresponding to the instruction and separator.
        # Then trunctate the inputs / pad the inputs according to args.max_length
        # Name the input as "input_ids"
        # 
        input_ids = instr_tokens["input_ids"] + sep_tokens + resp_tokens["input_ids"]
        labels = [-100] * len(instr_tokens["input_ids"]) + [-100] * len(sep_tokens) + resp_tokens["input_ids"]
        input_ids = input_ids[:args.max_length]
        labels = labels[:args.max_length]
        input_ids = input_ids + [tokenizer.pad_token_id] * (args.max_length - len(input_ids))
        labels = labels + [-100] * (args.max_length - len(labels))

        # TODO: Create attention mask
        attention_mask = [1 if token_id != tokenizer.pad_token_id else 0 for token_id in input_ids]

        # Your code ends here.

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    tokenized_dataset = dataset.map(tokenize_fn, batched=False)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    dataloader = DataLoader(tokenized_dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    samples = random.sample(range(len(tokenized_dataset)), 5)
  
    for epoch in range(args.epochs):

        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        print("\nSample generations:")
        for sample_index in samples:
            sample = dataset[sample_index]
            prompt = sample["instruction"]
            print(f"\nPrompt: {prompt}\n\n")
            # TODO: Generate a sentence using the input prompt 
            # Hint: Tokenize the prompt and then pass it to model.generate
            tokenized_prompt = tokenizer(prompt, return_tensors="pt").to(device)
            generated_ids = model.generate(
                tokenized_prompt["input_ids"],
                attention_mask=tokenized_prompt["attention_mask"],
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=50
            )
            generated_text = tokenizer.decode(generated_ids[0])
            print(f"Response: {generated_text[len(prompt):]}")
            # Your code ends here.
        
        total_loss = 0.0
        for index, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # TODO: Finish the main training loop
            # Hint: model.forward(...)
            # Make sure to divide the loss by the number of gradient accumulation steps
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / args.gradient_accumulation
            loss.backward()
            total_loss += loss.item()
            # Your code ends here.

            if (index+1) % args.gradient_accumulation == 0:
                # TODO: Perform gradient accumulation
                optimizer.step()
                optimizer.zero_grad()
                print(f"Batch {index+1}/{len(dataloader)}, Loss: {total_loss}")
                total_loss = 0.0
                # Your code ends here.

        # Handle any remaining gradients at the end of epoch
        if total_loss > 0:
            optimizer.step()
            optimizer.zero_grad()

        
        # Optional: Saving the model checkpoint at each epoch
        # model.save_pretrained(args.save_path)
        # tokenizer.save_pretrained(args.save_path)
    
    # again sample some examples
    print("\nSample generations after training:")
    for sample_index in samples:
        sample = dataset[sample_index]
        prompt = sample["instruction"]
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
    

    # Save the model and tokenizer, this allow us to use in further DPO step
    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    print("\nFine-tuning complete. Model saved to", args.save_path)

if __name__ == "__main__":
    main()
