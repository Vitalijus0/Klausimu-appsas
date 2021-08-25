import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
import streamlit as st

trained_model_path = 'model/'
trained_tokenizer = 'tokenizer/'

model = T5ForConditionalGeneration.from_pretrained(trained_model_path)
tokenizer = T5Tokenizer.from_pretrained(trained_tokenizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print ("device ",device)
model = model.to(device)


def main():
    st.title(" Atviri klausimai")

    st.write('Norėdami pasiūlyti pagalbą besimokant iš įvairių tekstų, pasitelkėme dirbtinį intelektą. Yra daug būdų mokytis informaciją iš teksto, o vienas iš pasitikrinti – pabandyti įrašyti praleistus esminius žodžius. Tiek dirbtinis intelektas, tiek mes mokomės teisingai suprasti lietuvių kalbą ir pateikti prasmę turinčius sakinius su trūkstamais žodžiais. Nors tai gali pasirodyti paprasta – iš tikrųjų tai yra gan sudėtinga semantinė užduotis. Prašome jūsų pagalbos testuojant prototipą')

    menu = ["Atviri klausimai", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    add_slider = st.sidebar.slider(
    'Nurodikyte atvirų klausimų skaičių',
    0, 10, (3)
)

    if choice == "Atviri klausimai":
        #Text area

        context  = st.text_area("Šiame lange nurodomas tekstas", height=300, max_chars=10000)

        answer = st.text_area("Šiame lange nurodomas teisingas atsakymas, kuriam algoritmas formuluos klausimą, vėliau šis procesas bus automatizuotas, suteikiant galimybę vartotojui pasirinkti raktažodžius", height=64, max_chars=512)

        if st.button("sukurti klausimus"):
            text = "context: "+context + " " + "answer: " + answer + " </s>"
            #print (text)

            encoding = tokenizer.encode_plus(text,max_length =512, padding=True, return_tensors="pt")
            #print (encoding.keys())

            input_ids,attention_mask  = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

            model.eval()
            beam_outputs = model.generate(
                input_ids=input_ids,attention_mask=attention_mask,
                max_length=72,
                early_stopping=True,
                num_beams=add_slider,
                num_return_sequences=add_slider

            )

            for beam_output in beam_outputs:
                sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
                st.write(sent[10:])




if __name__ == "__main__":
    main()