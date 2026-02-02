import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import pandas as pd
import numpy as np
import os
import random

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

class WassersteinMetric(nn.Module):
    
    def __init__(self):
        super(WassersteinMetric, self).__init__()

    def forwad(self,u_values, v_values, u_weights=None, v_weights=None):

    #def torch_wasserstein_distance(u_values, v_values, u_weights=None, v_weights=None):
        # NOTE Wasserstein Metrik
        # https://forkxz.github.io/blog/2024/Wasserstein/

        # Ensure that the input tensors are batched
        assert u_values.dim() == 2 and v_values.dim() == 2, "Input tensors must be 2-dimensional (batch_size, num_values)"

        batch_size, u_size = u_values.shape
        _, v_size = v_values.shape

        # Sort the values
        u_sorter = torch.argsort(u_values, dim=1)
        v_sorter = torch.argsort(v_values, dim=1)

        # Concatenate and sort all values for each batch
        all_values = torch.cat((u_values, v_values), dim=1)
        all_values, _ = torch.sort(all_values, dim=1)
        # Compute differences between successive values
        deltas = torch.diff(all_values, dim=1)

        # Get the respective positions of the values of u and v among the values of both distributions
        all_continue = all_values[:, :-1].contiguous()
        u_cdf_indices = torch.searchsorted(u_values.gather(1, u_sorter).contiguous(), all_continue, right=True)
        v_cdf_indices = torch.searchsorted(v_values.gather(1, v_sorter).contiguous(), all_continue, right=True)

        # Calculate the CDFs of u and v using their weights, if specified
        if u_weights is None:
            u_cdf = u_cdf_indices.float() / u_size
        else:
            u_sorted_cumweights = torch.cat((torch.zeros((batch_size, 1)), torch.cumsum(u_weights.gather(1, u_sorter), dim=1)), dim=1)
            u_cdf = u_sorted_cumweights.gather(1, u_cdf_indices) / u_sorted_cumweights[:, -1].unsqueeze(1)

        if v_weights is None:
            v_cdf = v_cdf_indices.float() / v_size
        else:
            v_sorted_cumweights = torch.cat((torch.zeros((batch_size, 1)), torch.cumsum(v_weights.gather(1, v_sorter), dim=1)), dim=1)
            v_cdf = v_sorted_cumweights.gather(1, v_cdf_indices) / v_sorted_cumweights[:, -1].unsqueeze(1)

        return torch.sum(torch.abs(u_cdf - v_cdf) * deltas, dim=1)

class Generator(nn.Module):
    def __init__(self, noise_dim, hidden_dim, output_size):
        super(Generator, self).__init__()
        self.l1 = nn.Linear(in_features=noise_dim, out_features=hidden_dim) 
        self.relu = nn.ReLU() 
        self.l2 = nn.Linear(in_features=hidden_dim, out_features=output_size) 

    def forward(self, x): 
        l1_out = self.l1(x) 
        relu_out = self.relu(l1_out) 
        l2_out = self.l2(relu_out) 
        return torch.tanh(l2_out) # NOTE Features wurden über standard Scaler normalisiert [-1,1] <-- tanh sorgt dafür das der Output auch zwischen -1 und 1 ist
    
    def generate_samples(self, n):
        pass

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes= 1):
        super(Discriminator, self).__init__() 
        self.l1 = nn.Linear(in_features=input_size, out_features=hidden_size) 
        self.relu = nn.LeakyReLU()
        self.l2 = nn.Linear(in_features=hidden_size, out_features=num_classes) 
    
    def forward(self, x): 
        l1_out = self.l1(x) 
        relu_out = self.relu(l1_out) 
        l2_out = self.l2(relu_out) 
        l2_out = torch.sigmoid(l2_out)
        return l2_out 

class DataPrep():
    def __init__(self, datafile, categorical_columns, noise_dim, value_filter=["."], ):
        self.categorical_columns = categorical_columns
        self.df = pd.read_csv(datafile)
        self.df.dropna(inplace=True)

        self.noise_dim = noise_dim
        self.full_noise_dim = None

        self.generator_features = len(self.df.columns)-len(self.categorical_columns)
        
        for filter_val in value_filter:
            for col in self.categorical_columns:
                values= self.df[self.df[col] == filter_val].index
                self.df.drop(values, inplace=True)

        self.df_count = self.df.groupby(self.categorical_columns).count().reset_index()

        num_combs = self.df_count.iloc[:,len(self.categorical_columns)+1].sum()
        self.df_count["probability"] = [x/num_combs for x in self.df_count.iloc[:,len(self.categorical_columns)+1]]
        
        self.__init_preprocessing_models()

    def __init_preprocessing_models(self):
        self.encoder_noise  = OneHotEncoder()
        self.collumn_trans  = ColumnTransformer(transformers=[("cat", OneHotEncoder(), self.categorical_columns)],remainder=StandardScaler())
        self.encoded_noisecondition_tensor = self.encoder_noise.fit_transform(self.df_count[self.categorical_columns]).toarray() 
        self.full_noise_dim = self.noise_dim + self.encoded_noisecondition_tensor.shape[1]

    def generate_training_test_data(self, boootstrap_multiplier=10, test_size=0.33, random_state=42):
        transformed = self.collumn_trans.fit_transform(self.df)
        #print("Transformed_Train", transformed)

        X = resample(transformed,replace=True,n_samples=boootstrap_multiplier,random_state=random_state) 

        self.total_features = X.shape[1] # Alle Spalten nach der Transformation
        X_train, X_test = train_test_split(X, test_size=test_size, random_state=random_state)
        return X_train, X_test
    

    def gen_noise_tensor(self, batch_size):
        cat = np.vstack(random.choices(self.encoded_noisecondition_tensor , weights=self.df_count["probability"], k=batch_size))#[0]
        num = torch.rand(batch_size, self.noise_dim)
        noise_tensor = torch.cat(tensors=(num,torch.from_numpy(cat)), dim=1)
        return noise_tensor, cat
        
    def rate_model(self, real_data, generated_data):
        wass_metric = WassersteinMetric()
        distance = wass_metric.forwad(real_data, generated_data)
        return distance

class CTGanTraining():
    def __init__(self, dataset, categorical_collumns, 
                 test_size=0.33,learning_rate = 0.01, 
                 random_state =42, batch_size=265, 
                 bootstrap_multiplier=10, numeric_noise_dim=128, epochs=1000):
        
        self.dataset = dataset
        self.categorical_collumns = categorical_collumns
        self.tes_size = test_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.batch_size = batch_size
        self.bootstrap_multiplier = bootstrap_multiplier
        self.numeric_noise_dim = numeric_noise_dim  
        self.epochs = epochs
        self.noise_dim = None
        self.X_train = None
        self.x_test = None     
        self.num_features = None 
        self.num_gen_features = None

    def __run_dataprep(self):
        self.dataprep = DataPrep(datafile=self.dataset, categorical_columns=self.categorical_collumns, noise_dim=self.numeric_noise_dim)
        self.dataprep.generate_training_test_data
        self.noise_dim = self.dataprep.full_noise_dim
        self.X_train, self.x_test = self.dataprep.generate_training_test_data()
        self.num_features = self.dataprep.total_features
        self.num_gen_features = self.dataprep.generator_features
    
    def _init_models(self):
        self.generator = Generator(self.dataprep.full_noise_dim,64, self.num_gen_features)
        self.discriminator=Discriminator(self.num_features,64)

        #criterion = nn.BCEWithLogitsLoss() # NOTE passt sonst nicht mit Sgimoid beim output layer 
        self.criterion = nn.BCELoss() 
        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=self.learning_rate, betas=(0.5, 0.999)) 
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))

    def _show_progress(self, epoch, real_loss, fake_loss, gen_loss):
        if epoch % 10 == 0:
            print(f'Epoch [{epoch+1}/{self.epochs}], ', 
                    f'Discriminator Loss: {real_loss.item() + fake_loss.item():.3f}, '
                    f'Generator Loss: {gen_loss.item():.3f}')

    def run_training(self):
        self.__run_dataprep()
       
       

        self._init_models()

        if type(self.X_train)!= None:
            for epoch in range(self.epochs):
                for i in range(0, len(self.X_train), self.batch_size):
                    real_data = torch.FloatTensor(self.X_train[i:i+self.batch_size])

                    # Train Discriminator
                    self.discriminator_optimizer.zero_grad()

                    # Real data
                    real_labels = torch.ones(real_data.size(0), 1)
                    real_outputs = self.discriminator(real_data)
                    real_loss = self.criterion(real_outputs, real_labels)

                    # Fake data
                    noise_tensor , cat_values = self.dataprep.gen_noise_tensor(batch_size=real_data.size(0))

                    #print(noise_tensor.shape, self.dataprep.full_noise_dim,64, self.num_gen_features)
                    fake_data = self.generator(noise_tensor.float())
                    #print("fakiout", fake_data)

                    fake_data = torch.cat(tensors=(torch.from_numpy(cat_values),fake_data), dim=1)

                    #print("dimensioncomp", fake_data.shape, real_data.shape, fake_data.dtype)
                    fake_labels = torch.zeros(real_data.size(0), 1)
                    fake_outputs = self.discriminator(fake_data.float().detach())
                    fake_loss = self.criterion(fake_outputs, fake_labels)

                    #print("here")

                    # Backprop and optimize
                    d_loss = real_loss + fake_loss
                    d_loss.backward()
                    self.discriminator_optimizer.step()

                    # Train Generator
                    self.generator_optimizer.zero_grad()
                    gen_labels = torch.ones(real_data.size(0), 1)  # We want the generator to fool the discriminator
                    gen_outputs = self.discriminator(fake_data.float())
                    gen_loss = self.criterion(gen_outputs, gen_labels)

                    # Backprop and optimize
                    gen_loss.backward()
                    self.generator_optimizer.step()
                self._show_progress(epoch, real_loss, fake_loss, gen_loss)
        else:
            print("No training data available. Please run data preparation first.")

if __name__ == "__main__":
    csvfile = os.path.join(os.getcwd(), "penguins_size.csv")
    #data_prep = DataPrep(datafile=csvfile, categorical_columns=["species","island","sex"])
    #data_prep.generate_training_test_data()
    #print(data_prep.gen_noise_tensor(noise_dim=128, batch_size=5))
    ctgan = CTGanTraining(dataset=csvfile,categorical_collumns=["species","island","sex"])
    ctgan.run_training()
    # TODO Dimensionen für Noise Tensor nochmal kontrollieren
    # TODO die ganze Zusammensetzung für die Fake Daten funktionieren noch nicht
    # Es muss noch die Zusammensetzung der ursprünglichen OneHot Anteile mit den vom Generator
    # generierten Skalierten Werten kombiniert werden 
    # TODO Trainingsablauf noch einmal mit gan_flowers.ipynb vergleichen
    # TODO Zusammensetzungen der Fake Daten überprüfen
    # TODO Labels Smoothing implementieren real_labels = torch.ones(self.batch_size, 1) * 0.9
    # sonst mode collapse
    # TODO eventuell Wasserstein Metrik zur Bewertung des Modells implementieren
    # TODO CSV Export der generierten Daten implementieren
    # TODO Export Generator Model nach dem Training
    """
     def generate(self, n_samples):
        with torch.no_grad():
            # Rauschen und Bedingungen generieren
            noise = self.dataprep.gen_noise_tensor(noise_dim=128, batch_size=n_samples)
            
            # Synthetische Daten erzeugen (Werte zwischen -1 und 1)
            fake_data_raw = self.generator(noise).numpy()
            
            # Rücktransformation in echte Werte
            # (Nutzt den ColumnTransformer aus deinem DataPrep)
            final_data = self.dataprep.collumn_trans.named_transformers_['remainder'].inverse_transform(
                fake_data_raw[:, len(self.dataprep.categorical_columns):] # Beispielhaft für die Scaler-Spalten
            )
            
            # Hier müsstest du noch die One-Hot-Spalten zurückrechnen
            return pd.DataFrame(fake_data_raw) # Vereinfachter Export
    """
    # Test 12