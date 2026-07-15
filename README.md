# ToSCA
## Total Scattering Conditioned Variational Autoencoder for Atomic Structure Prediction from Pair Distribution Function Data

#### Ulrik Friis-Jensen $^{1,2}$, Frederik L. Johansen $^{1,2}$, Erik B. Dam $^{2}$, Raghavendra Selvan $^{2}$, and Kirsten M. Ø. Jensen $^{1}$

1: Department of Chemistry & Nano-Science Center, University of Copenhagen <br>
2: Department of Computer Science, University of Copenhagen <br>

### Abstract
Designing new materials is often slow and experimentally intensive, with structural characterization representing a particular bottleneck. Recent advances in machine learning (ML) aim to accelerate this process by predicting plausible atomic structures describing a material directly from experimental data.
Here we present the total scattering conditioned variational autoencoder (ToSCA), a generative ML model for predicting plausible atomic structures of metal oxide nanomaterials from total scattering and pair distribution function (PDF) data. ToSCA learns a chemically meaningful latent representation and predicts unit cells within tolerances suitable for subsequent real-space refinement. The model also shows behavior consistent with learning structural relationships between related crystal types, and can correctly place the new CHILI-Interpolation dataset in the latent space.
Finally, we demonstrate that ToSCA is able to retain useful performance in a realistic use-case by testing it on 11 experimental PDFs from 3 crystal types, including test cases outside of the training distribution.
