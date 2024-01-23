import json
from transformers import AutoTokenizer, AutoModel

# Initialize tokenizer and model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

def create_embeddings(text):
    # Tokenize and create embeddings
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

def read_jsonl_and_create_embeddings(input_file_path):
    f = open(input_file_path)
    data = json.load(f)
    embeddings = []
    for i in data['items']:
        print(i)
        # Create embeddings for the company name
        company_name = i['company_name']
        company_name_embeddings = create_embeddings(company_name)
        # Save the item along with the new embedding property
        print('Vector Length: ' + str(company_name_embeddings.size))
        i['company_name_embedding'] = company_name_embeddings.tolist()
        embeddings.append(i)
 
    # Closing file
    f.close()
    
    return embeddings

def save_embeddings_to_jsonl(embeddings, output_file_path):
    with open(output_file_path, 'w') as file:
        # create the initial json object and add the items property
        file.write('{"items":[')
        for data in embeddings:
            json_line = json.dumps(data)
            # dont write a , if the entry is the last one
            if data == embeddings[-1]:
                file.write(json_line + '\n')
            else:
                file.write(json_line + ',\n')
            
        file.write(']}')

# File paths
input_file_path = './data/data.json'
output_file_path = './output/data.json'

# Process the file
embeddings = read_jsonl_and_create_embeddings(input_file_path)

# Save the embeddings
save_embeddings_to_jsonl(embeddings, output_file_path)
