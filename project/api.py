# ============================================================
# Run with: streamlit run api.py --server.fileWatcherType none
# ============================================================
import pickle
import torch
from torch import nn
import streamlit as st


class MyBaseModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accuracy_train_history = []
        self.accuracy_test_history = []
        self.loss_train_history = []
        self.loss_test_history = []
        self.train_time = 0

    def save(self, path):
        torch.save(self, path)

    @staticmethod
    def load(path) -> nn.Module|None:
        pass
    
    
features = 10000
class NeuralNetwork1(MyBaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(features, features*2//3),
            nn.ReLU(),
            nn.Linear(features*2//3, features*2//3//2),
            nn.ReLU(),
            nn.Linear(features*2//3//2, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits




def process(tensor, model):
    class_names = ['praise', 'amusement', 'anger', 'disapproval', 'confusion', 'interest', 'sadness', 'fear', 'joy', 'love']
    index = model(tensor).argmax().item()
    return class_names[index]

@st.cache_resource
def load_model(path):
    model = NeuralNetwork1()
    model = torch.load(path)
    model.eval()
    return model.to('cuda')

@st.cache_resource
def load_tensor_index(path):
    with open(path, 'rb') as f:
        tensor_index = pickle.load(f)
    return tensor_index

def pre_process(text, tensor_index):
    x_tensor = torch.zeros(1, len(tensor_index), dtype=torch.float16)
    for word in text.lower().split():
        if word in tensor_index:
            x_tensor[0, tensor_index[word]] += 1
    return x_tensor.to('cuda')


def main():
    model = load_model('project/models/nlp_model4.pt')
    tensor_index = load_tensor_index('project/models/tensor_index')
    st.title("Simple Streamlit App")
    user_input = st.text_input("Enter something:")
    user_input_tensor = pre_process(user_input, tensor_index)
    if st.button("Submit"):
        response = process(user_input_tensor, model)
        st.write(response)

if __name__ == "__main__":
    main()
