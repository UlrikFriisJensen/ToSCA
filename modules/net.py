#%% Imports

import torch

#%% Model

class SCVAE(torch.nn.Module):
    def __init__(self, encoder, decoder, classifier, latent_dim, n_classes):
        super(SCVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        self.latent_dim = latent_dim
        self.n_classes = n_classes

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return mu + eps*std
        else:
            return mu

    def forward(self, data):
        mu, logvar = self.encoder(data)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar, self.classifier(z)
    