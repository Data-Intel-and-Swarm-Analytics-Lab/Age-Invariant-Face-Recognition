# feature_extraction_and_classification.py

import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from transformers import ViTFeatureExtractor, ViTModel, ViTConfig
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import umap
from tqdm import tqdm


class ExtraTransformerBlock(nn.Module):
    """Defines an additional Transformer block to augment ViT."""
    def __init__(self, config):
        super(ExtraTransformerBlock, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

    def forward(self, hidden_states):
        return self.transformer_encoder(hidden_states)


class ModifiedViTModel(nn.Module):
    """Modified Vision Transformer with an extra Transformer block."""
    def __init__(self):
        super(ModifiedViTModel, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        config = ViTConfig.from_pretrained('google/vit-base-patch16-224-in21k')
        self.extra_block = ExtraTransformerBlock(config)

    def forward(self, inputs):
        outputs = self.vit(**inputs)
        last_hidden_state = outputs.last_hidden_state
        extra_features = self.extra_block(last_hidden_state)
        return extra_features


class FeatureExtractor:
    """Handles feature extraction methods (ViT, SIFT, LBP) and preprocessing."""
    def __init__(self):
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.vit_model = ModifiedViTModel()

    def extract_vit_features(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inputs = self.feature_extractor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = self.vit_model(inputs)
        return outputs.flatten().detach().numpy()

    @staticmethod
    def extract_sift_features(img):
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(img, None)
        if descriptors is None:
            descriptors = np.zeros((1, 128))
        return descriptors

    @staticmethod
    def extract_lbp_features(img):
        lbp = local_binary_pattern(img, P=8, R=1, method='uniform')
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)
        return hist

    @staticmethod
    def concatenate_features(sift_features, lbp_features, vit_features):
        sift_features = sift_features.flatten() if sift_features is not None else np.zeros(128)
        lbp_features = lbp_features.flatten() if lbp_features is not None else np.zeros(11)
        vit_features = vit_features.flatten() if vit_features is not None else np.zeros(768)

        combined_features = np.concatenate((sift_features, lbp_features, vit_features))
        fixed_length = 907
        if len(combined_features) < fixed_length:
            combined_features = np.concatenate((combined_features, np.zeros(fixed_length - len(combined_features))))
        elif len(combined_features) > fixed_length:
            combined_features = combined_features[:fixed_length]

        return combined_features


class DimensionalityReduction:
    """Applies dimensionality reduction techniques."""
    @staticmethod
    def apply_ipca(X, n_components=5, batch_size=500):
        ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
        return ipca.fit_transform(X)

    @staticmethod
    def apply_umap(X, n_components=10):
        umap_model = umap.UMAP(n_components=n_components)
        return umap_model.fit_transform(X)


class ModelTrainer:
    """Trains and evaluates machine learning models."""
    def __init__(self):
        self.models = {
            'RandomForest': RandomForestClassifier(),
            'KNN': KNeighborsClassifier(),
            'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        }

    def train_and_evaluate(self, X_train, X_test, y_train, y_test, method_name):
        results = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro')
            recall = recall_score(y_test, y_pred, average='macro')
            f1 = f1_score(y_test, y_pred, average='macro')

            results[name] = {
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1
            }

        return results
