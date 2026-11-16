# Variational Autoencoder (VAE): Reconstruction & Latent Space Interpolation

A convolutional **Variational Autoencoder** built in **PyTorch**, used in two experiments:

- **Part A:** train a VAE on a single image, then see how it reconstructs a completely different image
- **Part B:** train a VAE on two images together, then blend between them by interpolating in the latent space

---

## Repository Contents

| File | Description |
|---|---|
| `Part A` | Part A: single-image training and out-of-distribution reconstruction |
| `Part B` | Part B: joint training on two images and latent space interpolation |
| `train.jpeg` | Image 1: a golden retriever |
| `test.jpeg` | Image 2: a UChicago building |

## Model Architecture

Both parts use the same VAE (**~2.96M parameters**):

```
Input (3×64×64)
 → Encoder: 4 × [Conv 4×4, stride 2] + ReLU     3 → 32 → 64 → 128 → 256 channels  (64×64 → 4×4)
 → Flatten → two linear heads: μ and log σ²     (latent dim = 128)
 → Reparameterization: z = μ + σ · ε
 → Linear → reshape to 256×4×4
 → Decoder: 4 × [ConvTranspose 4×4, stride 2]   256 → 128 → 64 → 32 → 3 channels  (4×4 → 64×64)
 → Sigmoid
```

**Loss (ELBO):** sum-reduced MSE reconstruction loss plus KL divergence from the standard normal prior. Part B uses a β-weighted KL term.

## Part A: Single-Image VAE

**Setup:** the VAE is trained only on the dog image, then given the building image to reconstruct. 16 random samples are also drawn from the prior.

| Setting | Value |
|---|---|
| Image size | 64×64 |
| Latent dim | 128 |
| Epochs | 300 |
| Learning rate | 1e-3 (Adam) |
| KL weight | 1.0 |

**Results**
- The loss fell from **516 → 23** over 300 epochs.
- The reconstruction of the building **looks like the dog**: a blurry, slightly distorted golden retriever.
- All the random latent samples are also variations of the dog.

**Why:** the model has only seen one image, so its latent space only represents the dog. Any input, even a building, gets mapped to a nearby point in that space, and the decoder produces a dog. This shows a key limitation: a VAE can't reconstruct inputs outside its training distribution.

## Part B: Two-Image VAE with Latent Interpolation

**Setup:** both images are stacked into one batch so the VAE learns a **shared latent space**. Each image is encoded to its mean vector (μ₁, μ₂), and the decoder generates a 9-step path between them.

| Setting | Value |
|---|---|
| Image size | 64×64 |
| Latent dim | 128 |
| Epochs | 500 |
| Learning rate | 1e-3 (Adam) |
| KL weight (β) | 0.5 |
| Interpolation steps | 9 |

### Spherical Linear Interpolation (Slerp)
Plain linear interpolation gives uneven blends: the two latent vectors have different lengths, so the geometric midpoint isn't the visual midpoint, and one image dominates. Slerp moves at a constant angular speed along the arc between μ₁ and μ₂:

$$z_{\text{slerp}}(\alpha) = \frac{\sin((1-\alpha)\Omega)}{\sin\Omega}\,\hat{\mu}_1 + \frac{\sin(\alpha\,\Omega)}{\sin\Omega}\,\hat{\mu}_2$$

where Ω is the angle between the two vectors. The result is rescaled to the interpolated length, and VAE noise is added (`z = z_slerp + σ(α)·ε`), so re-running the cell gives a slightly different variation of the same blend.

**Results**
- The loss fell from **1276 → 47** over 500 epochs. It starts higher than in Part A because the model is learning two images.
- The strip moves smoothly from the dog (α = 0) to the building (α = 1), with a real 50/50 blend at α = 0.5.
- Random prior samples still lean toward the dog, since both images share warm tones and green backgrounds.

## Key Takeaway

| | Part A | Part B |
|---|---|---|
| Training data | Dog only | Dog + building |
| Output for the building | A dog | A faithful endpoint of the interpolation |
| What it shows | The limits of a narrow training distribution | How a shared latent space lets you generate meaningful in-between images |
