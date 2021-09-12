

from flask import render_template, request, Flask


from flask import Flask, render_template, g, request, session, redirect, url_for, jsonify

from flask import render_template_string
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer

trained_model_path = 'model/'
trained_tokenizer = 'tokenizer/'

model = T5ForConditionalGeneration.from_pretrained(trained_model_path)
tokenizer = T5Tokenizer.from_pretrained(trained_tokenizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = model.to(device)

app = Flask(__name__)


@app.route('/',methods=['GET', 'POST'])
def home():
    
    if request.method == 'POST':
        print("ok")
        if request.form.get("question_gen"):

            default_value = ' '
            context = str(request.form.get('context', default_value))
            print(context)

            text2 = str(request.form.get('answer', default_value))
            
            answers = text2.split('//')
            print('===========')
            print(answers)
            print('===========')
            #print(request.form.get('number_of_quest'))
            #print('===================')
            #klausimu_sk = int(request.form.get('number_of_quest'))
            klausimu_sk = 5

            #context ="Vilnius – Lietuvos sostinė ir didžiausias šalies miestas, Vilniaus apskrities, rajono ir miesto savivaldybės centras."
            #answer = "Donald Trump"
            atviri_klausimai = []
            for answer in answers:
                text = "context: "+context + " " + "answer: " + answer + " </s>"

                encoding = tokenizer.encode_plus(text,max_length =512, padding=True, return_tensors="pt")
                #print (encoding.keys())
                input_ids,attention_mask  = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

                model.eval()
                beam_outputs = model.generate(
                    input_ids=input_ids,attention_mask=attention_mask,
                    max_length=72,
                    early_stopping=True,
                    num_beams=klausimu_sk,
                    num_return_sequences=klausimu_sk
                )
                
                
                for beam_output in beam_outputs:
                    sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
                    atviri_klausimai.append(sent[10:])
                    print (sent)



    elif request.method == 'GET':
        
        atviri_klausimai = ''
        
    
    return render_template('test.html', atviri_klausimai = atviri_klausimai)


if __name__ == '__main__':
    app.run(debug=True)