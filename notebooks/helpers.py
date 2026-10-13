import numpy as np
import torch
import os
from legacy_kl import computeKL
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ndcg_score

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
DB_FILE   = 'FairCVdb.npy'
DEVICE    = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Inspired by Swati's code
def build_features(debias ='none'):
    fairCV = np.load(os.path.join(DATA_PATH, DB_FILE), allow_pickle = True).item()
    profiles_train = fairCV['Profiles Train']
    profiles_test = fairCV['Profiles Test']

    labels_train_biased = fairCV['Biased Labels Train (Gender)']
    labels_train_blind = fairCV['Blind Labels Train']

    labels_test_biased = fairCV['Biased Labels Test (Gender)']
    labels_test_blind = fairCV['Blind Labels Test']


    if debias == 'none':
        # TABULAR ORIGINAL
        tabular_train = profiles_train[:, [1, 4, 5, 6, 7, 8, 9, 10]]
        tabular_test = profiles_test[:, [1, 4, 5, 6, 7, 8, 9, 10]]

        # TEXT ORIGINAL
        text_train = np.load(os.path.join(DATA_PATH, 'original_bios_train_bert.npy'))
        text_test  = np.load(os.path.join(DATA_PATH, 'original_bios_test_bert.npy'))

        # IMAGES ORIGINAL
        faces_train = profiles_train[:, 11:31]
        faces_test = profiles_test[:, 11:31]

    elif debias == 'tabular':
        # TABULAR BLINDED
        tabular_train = profiles_train[:, 4:11]
        tabular_test = profiles_test[:, 4:11]

        # TEXT ORIGINAL
        text_train = np.load(os.path.join(DATA_PATH, 'original_bios_train_bert.npy'))
        text_test  = np.load(os.path.join(DATA_PATH, 'original_bios_test_bert.npy'))

        # IMAGES ORIGINAL
        faces_train = profiles_train[:, 11:31]
        faces_test = profiles_test[:, 11:31]

    elif debias == 'tabular-text':
        # TABULAR BLINDED
        tabular_train = profiles_train[:, 4:11]
        tabular_test = profiles_test[:, 4:11]

        # TEXT BLINDED
        text_train  = np.load(os.path.join(DATA_PATH, 'blinded_bios_train_bert.npy'))
        text_test   = np.load(os.path.join(DATA_PATH, 'blinded_bios_test_bert.npy'))

        # IMAGES ORIGINAL
        faces_train = profiles_train[:, 11:31]
        faces_test = profiles_test[:, 11:31]

    elif debias == 'tabular-image':
        # TABULAR BLINDED
        tabular_train = profiles_train[:, 4:11]
        tabular_test = profiles_test[:, 4:11]

        # TEXT ORIGINAL
        text_train = np.load(os.path.join(DATA_PATH, 'original_bios_train_bert.npy'))
        text_test  = np.load(os.path.join(DATA_PATH, 'original_bios_test_bert.npy'))

        # IMAGES AGNOSTIC
        faces_train = profiles_train[:, 31:51]
        faces_test = profiles_test[:, 31:51]

    elif debias == 'tabular-image-text':
        # TABULAR BLINDED
        tabular_train = profiles_train[:, 4:11]
        tabular_test = profiles_test[:, 4:11]

        # TEXT BLINDED
        text_train  = np.load(os.path.join(DATA_PATH, 'blinded_bios_train_bert.npy'))
        text_test   = np.load(os.path.join(DATA_PATH, 'blinded_bios_test_bert.npy'))

        # IMAGES AGNOSTIC
        faces_train = profiles_train[:, 31:51]
        faces_test = profiles_test[:, 31:51]

    scaler = StandardScaler()
    tabular_train_scaled = scaler.fit_transform(tabular_train)
    tabular_test_scaled = scaler.transform(tabular_test)

    text_train_scaled = scaler.fit_transform(text_train)
    text_test_scaled = scaler.transform(text_test)

    faces_train_scaled = scaler.fit_transform(faces_train)
    faces_test_scaled = scaler.transform(faces_test)

    return (labels_train_biased, labels_train_blind, labels_test_biased, labels_test_blind,
            tabular_train_scaled, tabular_test_scaled, text_train_scaled, text_test_scaled, faces_train_scaled, faces_test_scaled)

def train_and_evaluate(model_fn, train_inputs, test_inputs, labels_train, seed, epochs=16, batch_size=128):
    """Train one model and return its predictions on the test set. Adapted for all three fusion strategies


    :param model_fn: a function that builds and returns the model, not the model (because we need to seed)
    :param train_inputs: a list of numpy arrays, one per modality, e.g. [tabular_train, text_train, faces_train]
    :param labels_train: training targets, we reshape this to match the model's input
    :return: (predictions as flat array, the trained model)
    """

    torch.manual_seed(seed)
    model = model_fn().to(DEVICE)

    # turn embeddings into tensors
    train_tensors = [torch.FloatTensor(x).to(DEVICE) for x in train_inputs]
    test_tensors  = [torch.FloatTensor(x).to(DEVICE) for x in test_inputs]
    labels = torch.FloatTensor(labels_train).reshape(-1, 1).to(DEVICE) # labels on same device as predictions

    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.L1Loss()

    for epoch in range(epochs):
        model.train()

        for i in range(0, len(labels), batch_size):
            batch = [t[i:i + batch_size] for t in train_tensors]
            predictions = model(*batch) # *batch means spread out as different arguments
            loss = criterion(predictions, labels[i:i + batch_size])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        predictions = model(*test_tensors).cpu().numpy().flatten()

    return predictions, model

def load_gender_test():
    fairCV = np.load(os.path.join(DATA_PATH, DB_FILE), allow_pickle=True).item()
    return fairCV['Profiles Test'][:, 1]


# EVALUATION METRICS

# Legacy KL formula is reimplemented separately; upstream source is not redistributed.
# Compute the KL divergence

    # The KL divergence is defined if P and Q are normalized and Q!=0 where P !=0

# Compute distributions from the raw scores, and return the KL divergence

def computeMAE(x,y):
    return np.mean(np.abs(x-y))


def computeDemographicParity(predictions, gender):
    scores_sorted = np.argsort(predictions)
    scores_sorted = scores_sorted[::-1]

    gender = gender[scores_sorted]
    top100 = gender[:100]

    top100_female = np.sum(top100)/100
    top100_male = 1 - top100_female

    p_val = min(top100_female/top100_male, top100_male/top100_female)

    return p_val, top100_female


# p = 75th percentile
def computeEqualityOfOpportunity(predictions, gender, labels_train_blind, labels_test_blind, verbose = False, p = 75):

    threshold = np.percentile(labels_train_blind, p)

    # copies, so we don't mutate the parameters
    labels_test_blind = labels_test_blind.copy()
    predictions = predictions.copy()

    labels_test_blind[labels_test_blind < threshold] = 0
    labels_test_blind[labels_test_blind >= threshold] = 1
    predictions[predictions < threshold] = 0
    predictions[predictions > threshold] = 1

    male = (gender == 0)
    female = (gender == 1)

    labels_male = (labels_test_blind[male] == 1)
    labels_female = (labels_test_blind[female] == 1)
    scores_male = predictions[male]
    scores_female = predictions[female]

    tpr_male = np.sum(scores_male[labels_male])/np.sum(labels_male)
    tpr_female = np.sum(scores_female[labels_female])/np.sum(labels_female)

    n_selected = predictions.sum()

    if verbose:
        print("Qualified males:", np.sum(labels_male))
        print("Qualified females:", np.sum(labels_female))
        print("Qualified selected (male):", np.sum(scores_male[labels_male]))
        print("Qualified selected (female):", np.sum(scores_female[labels_female]))
        print("Male TPR:", tpr_male)
        print("Female TPR:", tpr_female)
        print("Total selected:", n_selected)

    return (tpr_male - tpr_female), n_selected


def computeNDCG(predictions, labels_blind, k=10):

    # reshape because ndcg expects 2D
    return ndcg_score(labels_blind.reshape(1,-1), predictions.reshape(1,-1), k=k)
