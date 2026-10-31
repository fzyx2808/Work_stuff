import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from keras.datasets import imdb

#DATA
(train_data, train_labels), (test_data, test_labels) =imdb.load_data(num_words=10000)
#BLBNUTÍ
word_index = imdb.get_word_index()

seznam =[]
for (key, value) in word_index.items():
    seznam.append((value, key))
reversed = dict(seznam)

odkodovani = " ".join([reversed.get(i-3,"?") for i in train_data[0]])
#print(odkodovani)

#CHYSTÁNÍ DAT
def vektorizace(sekvence, dimenze=10000):
    vysledek = torch.zeros(len(sekvence),dimenze)
    for i, recenze in enumerate(sekvence):
        vysledek[i, recenze] = 1.0  # na pozice slov dá jedničky
    return vysledek
x_train = vektorizace(train_data)
x_test = vektorizace(test_data)

y_train = torch.tensor(train_labels, dtype=torch.float32)
y_test = torch.tensor(test_labels, dtype=torch.float32)

#Síť
class Klasifikator(nn.Module): #nn.Module dědí z nn.Module
    def __init__(self):
        super().__init__() # Zavolá konstruktor rodičovské třídy nn.Module
        self.fc1 = nn.Linear(10000,16)
        self.fc2 = nn.Linear(16,16)
        self.fc3 = nn.Linear(16,1)    #fc znamená fully connected

    def forward(self, x): # x je vstupní dávka dat
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x 

model = Klasifikator()

optimizer =torch.optim.Adam(model.parameters(), lr =0.001) #lr = learning_rate
loss_fn = nn.BCELoss() # loss="binary_crossentropy"

def presnost(y_model, y_true):
    predikce = (y_model >= 0.5).float() #vysledkem je tenzor typu bool
    return (predikce==y_true).float().mean().item() #mean() udělá pruměr a item() vytáhne toto číslo

#Validační  data
x_v, x_tr = x_train[:10000], x_train[10000:]
y_v, y_tr = y_train[:10000], y_train[10000:]


#TRENOVANI
# DataLoader je podavač dat: vezme dataset a při každém kroku smyčky vám podá jednu dávku (batch)
Train_loader = DataLoader(TensorDataset(x_tr,y_tr), batch_size=512, shuffle=True) #TensorDataset zajišťuje, že recenze a její štítek zůstanou vždy spolu, shuffle zajistí, že na začátku každé epochy se pořadí vzorků náhodně zamíchá.
Val_loader = DataLoader(TensorDataset(x_v,y_v), batch_size=512) # Validační data se nemíchají: slouží jen k měření a výsledek na pořadí nezávisí.

history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

for epocha in range(4):
    model.train()       # Přepne model do trénovacího režimu
    soucet_loss, spravne, celkem = 0.0, 0.0, 0.0
    for x,y in Train_loader:
        optimizer.zero_grad()
        y_pred = model(x).view(-1) # poslední dávka již nemá, resp. nemusí mít 512, view(-1) říká, dopočítej to sám a dej je do řady
        loss = loss_fn(y_pred,y)
        loss.backward()            # zjisti, jakým směrem upravit váhy.
        optimizer.step()           # upraví je

        soucet_loss += loss.item() * len(y)
        spravne += ((y_pred >= 0.5).float() == y).sum().item()
        celkem += len(y) 

    history["loss"].append(soucet_loss / celkem)
    history["accuracy"].append(spravne / celkem)

    model.eval()
    soucet_loss, spravne, celkem = 0.0, 0.0, 0.0
    with torch.no_grad(): 
         for x,y in Val_loader:
             y_pred = model(x).view(-1)
             soucet_loss += loss_fn(y_pred, y).item() * len(y)
             spravne += ((y_pred >= 0.5).float() == y).sum().item()
             celkem += len(y)      

    history["val_loss"].append(soucet_loss / celkem)
    history["val_accuracy"].append(spravne / celkem)

    print(f"Epocha {epocha + 1:2d}/20 | "
          f"loss {history['loss'][-1]:.4f} | acc {history['accuracy'][-1]:.4f} | "
          f"val_loss {history['val_loss'][-1]:.4f} | val_acc {history['val_accuracy'][-1]:.4f}")

# =============================================
#Finální testování
#==============================================

test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=512)

model.eval()
spravne, celkem = 0.0, 0.0
with torch.no_grad():
    for x,y in test_loader:
        y_pred = model(x).view(-1)
        spravne += ((y_pred >= 0.5).float() == y).sum().item()
        celkem += len(y)

uspesnost = spravne / celkem
print(f"Finální úspěšnost sítě je {uspesnost:.2%}")