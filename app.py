
import os, json, torch, requests
from flask import Flask, request
from transformers import GPTNeoForCausalLM, AutoTokenizer

# Manual set env keys to test with:
os.environ['Number_Of_Words'] = '400'
os.environ['AI_Model'] = 'gpt-neo-125M'
os.environ['Destination'] = ''
os.environ['Tags'] = 'Family Feud'
os.environ['Categories'] = 'AI Generated'


# Load the model into mem once then reuse as needed
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/" + os.getenv('AI_Model'))
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/" + os.getenv('AI_Model'))

question = "What is dance?"
answer = ''

def get_answer(question, model, tokenizer):
    input_ids = tokenizer(question, return_tensors="pt").input_ids

    # add the length of the prompt tokens to match with the mesh-tf generation
    max_length = int(os.getenv('Number_Of_Words')) + input_ids.shape[1] 

    # Run on GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gen_tokens = model.generate(
        input_ids,
        do_sample = True,
        min_length = max_length,
        max_length = max_length,
        temperature = .9,
        pad_token_id = tokenizer.eos_token_id,
        device = device,
    )

    return tokenizer.batch_decode(gen_tokens)[0]

def question_cleanup(question):
    # Remove non ASCII
    question = "".join(_ for _ in question if ord(_) < 126 and ord(_) > 31)

    # Trim the question of white spaces
    question = question.strip()

    # Convert the question into the proper capitalization.
    question = question.capitalize()
    
    # Add period if missing puncation
    letters = 'abcdefghijklmnopqrstuvwxyz'
    letters = [char for char in letters]

    if question[-1] in letters:
        question += '.'

    return question

def answer_cleanup(answer, question):
    # Trim everything after the final period
    answer = answer[0:answer.rfind('.')]
    answer += '.'

    # Remove the question from the answer
    # AI tends to repeat the question as part of the answer
    answer = answer.replace(question, '', 1)
    answer = answer.strip()

    # Add read more splice for Wordpress
    answer = answer[:300] + '<!--more-->' + answer[300:]

    # Put the output in quotes
    answer = '<blockquote>' + answer + '</blockquote>'

    # Add disclaimer
    answer += 'This page content has been unsupervised ' \
        'auto generated using ' +  os.getenv('AI_Model') + '. ' \
        'The content created is fictitious and describes a universe that does not exist.'
    return answer

#?question=alex
app = Flask(__name__)
@app.route('/', methods = ['GET'])
def index():
    question = request.args.get('question')
    if question is None:
        print('🕑 Waiting for a question to be asked')
        return '🕑 Waiting for a question to be asked'
    print(f'❓ Received question: {question}')

    answer = get_answer(question, model, tokenizer)
    answer = answer_cleanup(answer, question)
    print('✔️ Answer was recieved')

    tags = os.environ['Tags'] + ':' + os.environ['AI_Model']
    tags = tags.split(':')
    categories = os.environ['Categories']
    categories = categories.split(':')

    url = os.environ['Destination']  + '/json'
    json_data = { 
        "Title": question, 
        "Content": answer, 
        "Tags": tags,
        "Categories": categories,
    } 
    headers = {'Content-Type': 'application/json'}

    # Send the test request to the page
    requests.post(url, data=json.dumps(json_data), headers=headers)
    print('📨 Sent to WordPress to be added')

    return answer

if __name__ == '__main__':
    app.run(host= '0.0.0.0', debug=True, port=80)
