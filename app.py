import streamlit as st

import numpy as np
import time
import torch
from torch import nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_text(model,itos,stoi, block_size, max_len, start_str = None):
    g = torch.Generator()
    g.manual_seed(42)
    context = [0]*block_size
    if start_str:
        for char in start_str:
            context = context[1:] + [stoi[char]]
    text = start_str if start_str else ''
    for _ in range(max_len):
        x = torch.tensor(context).view(1,-1).to(device)
        y_pred = model(x)
        ix = torch.distributions.Categorical(logits=y_pred).sample().item()
        ch = itos[ix]
        text += ch
        context = context[1:] + [ix]
    return text

def type_text(text):
    text_element = st.empty()
    s = ''
    for i in text:
        s += i
        text_element.write(s + '$I$')
        time.sleep(0.005)
    text_element.write(s)

class NextChar(nn.Module):
    def __init__(self, block_size, vocab_size, emb_dim, hidden_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lin1 = nn.Linear(block_size * emb_dim, hidden_size)
        self.activation = nn.ReLU()  # Adding ReLU activation function
        self.dropout = nn.Dropout(0.2)  # Adding dropout with 20% probability
        self.lin2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.emb(x)
        x = x.view(x.shape[0], -1)
        x = self.activation(self.lin1(x))  # Applying ReLU activation
        x = self.dropout(x)  # Applying dropout
        x = self.lin2(x)
        return x

st.write("# Neural Network based Text Generator")

st.sidebar.title("Model Information")

st.sidebar.write("This is a simple Neural Network based text generator. It can predict next characters of the input text provided. The model uses lower case letters, punctuation marks and fullstops. The text is generated paragraph wise, because the model learnt this from the text corpus.")
corpus=st.selectbox(
    'Select Corpus',
    ('Startup Funding', 'Shakespear','Algebra latex Report')
)

num_chars = st.slider("Number of characters to be generated", 100, 5000, 1000)


# model = torch.compile(model)
if(corpus == 'Startup Funding'):
    with open("text files/startup_funding.txt", 'r') as file:
        thefile = file.read()
    sort_char = ''
    for char in thefile:
        if char in ['\x0c','\ufeff','»', '¿', 'â', 'ï', '”','€']:
            continue
        sort_char += char.lower()

    characters = sorted(list(set(sort_char)))
    
    stoi = {s: i+1 for i, s in enumerate(characters)}
    stoi['~'] = 0 # for padding
    stoi['—'] = 53
    # print(stoi)
    itos = {i : s for s,i in stoi.items()}
    
    col1, col2 = st.columns(2)
    with col1:
        option1 = st.selectbox('Select Block Size', ['10', '100'])

    with col2:
        option2 = st.selectbox('Select Embedding', ['60', '150'])
    if(option1 == '10' and option2 == '60'):
        block_size = 10
        emb_dim=60
    elif(option1 == '10' and option2 == '150'):
        block_size = 10
        emb_dim=150
    elif(option1 == '100' and option2 == '60'):
        block_size = 100
        emb_dim=60
    else:
        block_size = 100
        emb_dim=150
    input_text = st.text_input("Enter Text", placeholder="Enter a text (In Lower Case) or leave it empty")
    # lower the text
    model = NextChar(block_size, len(stoi), emb_dim, 512).to(device)
    # if input text contains any character other than the ones in the model, then throw an error
    for char in input_text:
        if char not in characters:
            st.error("Invalid Characters in the input text")
            break
    btn=st.button("Generate Text")
    if btn:
        st.subheader("Seed Text")
        type_text(input_text)
        if(option1 == '10' and option2 == '60'):
            model.load_state_dict(torch.load("models/startup_funding model/model_story11.pth", map_location = device))
        elif(option1 == '10' and option2 == '150'):
            model.load_state_dict(torch.load("models/startup_funding model/model_story12.pth", map_location = device))
        elif(option1 == '100' and option2 == '60'):
            model.load_state_dict(torch.load("models/startup_funding model/model_story21.pth", map_location = device))
        else:
            model.load_state_dict(torch.load("models/startup_funding model/model_story22.pth", map_location =device))
            
        gen_text = generate_text(model, itos, stoi, block_size, num_chars, input_text)
        st.subheader("Generated Text")
        # print(gen_text)
        type_text(gen_text)
    
elif(corpus == 'Algebra latex Report'):
    with open("text files/algerbra_latex_report.txt", 'r') as file:
        thefile = file.read()
    sort_char = ''
    for char in thefile:
        if char in ['\x0c','\ufeff','»', '¿', 'â', 'ï', '”','€']:
            continue
        sort_char += char.lower()

    characters = sorted(list(set(sort_char)))
    
    stoi = {s: i+1 for i, s in enumerate(characters)}
    stoi['~'] = 0 # for padding
    # stoi['—'] = 53
    # print(stoi)
    itos = {i : s for s,i in stoi.items()}
    # side by side select box for block size and embedding size
    col1, col2 = st.columns(2)
    with col1:
        option1 = st.selectbox('Select Block Size', ['10','25','50'])

    with col2:
        option2 = st.selectbox('Select Embedding', ['60', '150'])
    if(option1 == '10' and option2 == '60'):
        block_size = 10
        emb_dim=60
    elif(option1 == '10' and option2 == '150'):
        block_size = 10
        emb_dim=150
    elif(option1 == '25' and option2 == '60'):
        block_size = 25
        emb_dim=60
    elif(option1 == 25 and option2 == '150'):
        block_size ='25'
        emb_dim=150
    elif(option1 == '50' and option2 == '60'):
        block_size = 50
        emb_dim=60
    elif(option1 == '50' and option2 == '150'):
        block_size = 50
        emb_dim=150
    
    model_option = st.radio("Select Model?", ("Trained", "Untrained"))
    input_text = st.text_input("Enter Text", placeholder="Enter a text (In Lower Case) or leave it empty")
    # lower the text
    model = NextChar(block_size, len(stoi), emb_dim, 512).to(device)
    # if input text contains any character other than the ones in the model, then throw an error
    for char in input_text:
        if char not in characters:
            st.error("Invalid Characters in the input text")
            break
    btn=st.button("Generate Text")
    if btn:
        st.subheader("Seed Text")
        type_text(input_text)
        if model_option == 'Trained':
            if(option1 == '10' and option2 == '60'):
                model.load_state_dict(torch.load("models/algebra letex_code model/algebra10_60.pth", map_location = device))
            elif(option1 == '10' and option2 == '150'):
                model.load_state_dict(torch.load("models/algebra letex_code model/algebra10_150.pth", map_location = device))
            elif(option1 == '25' and option2 == '60'):
                model.load_state_dict(torch.load("models/algebra letex_code model/algebra25_60.pth", map_location = device))
            elif(option1 == '25' and option2 == '150'):
                model.load_state_dict(torch.load("models/algebra letex_code model/algebra25_150.pth", map_location = device))
            elif(option1 == '50' and option2 == '60'):
                model.load_state_dict(torch.load("models/algebra letex_code model/algebra50_60.pth", map_location = device))
            else:
                model.load_state_dict(torch.load("models/algebra letex_code model/algebra50_150.pth", map_location =device))
        else:
            model.load_state_dict(torch.load("models/algebra letex_code model/algebra_untrained.pth", map_location =device))
        
        gen_text = generate_text(model, itos, stoi, block_size, num_chars, input_text)
        st.subheader("Generated Text")
        # print(gen_text)
        type_text(gen_text)
        
        
elif(corpus == 'Shakespear'):
    with open("text files/shakespear.txt", 'r') as file:
        thefile = file.read()
    sort_char = ''
    for char in thefile:
        if char in ['\x0c','\ufeff','»', '¿', 'â', 'ï', '”','€']:
            continue
        sort_char += char.lower()

    characters = sorted(list(set(sort_char)))
    
    stoi = {s: i+1 for i, s in enumerate(characters)}
    stoi['~'] = 0 # for padding
    # print(stoi)
    itos = {i : s for s,i in stoi.items()}
    
    col1, col2 = st.columns(2)
    with col1:
        option1 = st.selectbox('Select Block Size', ['10','25','50', '100'])

    with col2:
        option2 = st.selectbox('Select Embedding', ['60', '150'])
    if(option1 == '10' and option2 == '60'):
        block_size = 10
        emb_dim=60
    elif(option1 == '10' and option2 == '150'):
        block_size = 10
        emb_dim=150
    elif(option1 == '25' and option2 == '60'):
        block_size = 25
        emb_dim=60
    elif(option1 == '25' and option2 == '150'):
        block_size = 25
        emb_dim=150
    elif(option1 == '50' and option2 == '60'):
        block_size = 50
        emb_dim=60
    elif(option1 == '50' and option2 == '150'):
        block_size = 50
        emb_dim=150
    elif(option1 == '100' and option2 == '60'):
        block_size = 100
        emb_dim=60
    else:
        block_size = 100
        emb_dim=150
    # Add radio button for selecting the model for trained and untrained model
    model_option = st.radio("Select Model?", ("Trained", "Untrained"))
    
    input_text = st.text_input("Enter Text", placeholder="Enter a text (In Lower Case) or leave it empty")
    # lower the text
    model = NextChar(block_size, len(stoi), emb_dim, 512).to(device)
    # if input text contains any character other than the ones in the model, then throw an error
    for char in input_text:
        if char not in characters:
            st.error("Invalid Characters in the input text")
            break
    btn=st.button("Generate Text")
    if btn:
        st.subheader("Seed Text")
        type_text(input_text)
        if model_option == 'Trained':
            if(option1 == '10' and option2 == '60'):
                model.load_state_dict(torch.load("models/shakespear model/shakespear10_60.pth", map_location = device))
            elif(option1 == '10' and option2 == '150'):
                model.load_state_dict(torch.load("models/shakespear model/shakespear10_150.pth", map_location = device))
            elif(option1 == '25' and option2 == '60'):
                model.load_state_dict(torch.load("models/shakespear model/shakespear25_60.pth", map_location = device))
            elif(option1 == '25' and option2 == '150'):
                model.load_state_dict(torch.load("models/shakespear model/shakespear25_150.pth", map_location = device))
            elif(option1 == '50' and option2 == '60'):
                model.load_state_dict(torch.load("models/shakespear model/shakespear50_60.pth", map_location = device))
            elif(option1 == '50' and option2 == '150'):
                model.load_state_dict(torch.load("models/shakespear model/shakespear50_150.pth", map_location = device))
            elif(option1 == '100' and option2 == '60'):
                model.load_state_dict(torch.load("models/shakespear model/shakespear100_60.pth", map_location = device))
            else:
                model.load_state_dict(torch.load("models/shakespear model/shakespear100_150.pth", map_location =device))
        else:
            model.load_state_dict(torch.load("models/shakespear model/shakespear_untrained.pth", map_location =device))
                
        gen_text = generate_text(model, itos, stoi, block_size, num_chars, input_text)
        st.subheader("Generated Text")
        # print(gen_text)
        type_text(gen_text)
