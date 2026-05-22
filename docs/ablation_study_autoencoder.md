# Ablation Study Result



## LD4LG Compression Network
The original formula of the LD4LG compression is structured

$Z = Z + MHA(q=Z, kv=[D(w);Z])$

$Z = FFN(Z)$ 

in which the Z is a latent learnable feature of the decoder network `D` output. This is then feed into an FFn to recast the output vector into desired shape. During study, sepcifclally training the autoencoder to reconstruct the GSM8K QA text, the performance of the network is quite weak
| step | Loss | lr |
|------|------|----|
| 1000 | 4.4786 | 1.00e-04 |
| 2000 | 2.9072 | 1.00e-04 |
| 3000 | 2.6166 | 1.00e-04 |
| 4000 | 2.4800 | 1.00e-04 |
| 5000 | 2.1753 | 1.00e-04 |

this is compared to separated atten implementation

$Z = Z + selfAtten(q=z, k=z, v=z)$

$Z = Z + corssAttn(q=z, k=E(w), v=E(w))$

$z = z + FFN(z)$
which produces a vastly superior result
| step | Loss | lr |
|------|------|----|
| 1000 | 2.4083 | 1.00e-04 |
| 2000 | 1.1186 | 1.00e-04 |
| 3000 | 0.8974 | 1.00e-04 |
| 4000 | 0.4249 | 1.00e-04 |
| 5000 | 0.3777 | 1.00e-04 |

is possible that the described architecture is more suitable for large amount of text training instead of the GSM8K data.