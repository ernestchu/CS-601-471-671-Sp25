from openai import OpenAI
from datasets import load_dataset

num_demos = 8
num_tests = 30

def prompt_template(passage, question):
    return f"{passage}. {question}?"

def get_even_data(dataset, num_examples, index):
    examples = []
    pos_count = 0
    neg_count = 0
    while pos_count < num_examples / 2 or neg_count < num_examples / 2:
        data = dataset[index]
        index += 1
        question = data['question']
        passage = data['passage']
        answer = data['answer']

        if answer:
            if pos_count >= num_examples / 2:
                continue
            pos_count += 1
        else:
            if neg_count >= num_examples / 2:
                continue
            neg_count += 1
        examples.append([prompt_template(passage, question), 'Yes' if answer else 'No'])
    return examples, index

print("Loading the dataset ...")
dataset = load_dataset("boolq")
dataset = dataset.shuffle()['train']  # shuffle the data

index = 0
demos, index = get_even_data(dataset, num_demos, index)
tests, index = get_even_data(dataset, num_tests, index)

demos = '\n\n'.join([f"{demo[0]}\n{demo[1]}" for demo in demos])

print("Starting analysis with OpenAI API ...")
# Given that we specify logit_bias of 100 for the tokens `Yes' and `No', we assume GPT would only response with `Yes' or `No'.
# A more robust implementation would require inferring the probability of `Yes' or `No' as the next predicted token, but that takes a lot more work ...
# So let's try the simpler approach for now!
print()
print("GT\tGPT")

client = OpenAI()
results = []

for test in tests:
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"{demos}\n\n{test[0]}\n",
        }],
        max_tokens=1,
        logit_bias={13022: 100, 3160: 100},
    )

    print(test[1], completion.choices[0].message.content, sep='\t')
    results.append(test[1] == completion.choices[0].message.content)

print()
print(f"Accuracy: {sum(results) / len(results)}")


