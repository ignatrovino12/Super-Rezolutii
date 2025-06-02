
# Super-Rezolutii

## Descrierea Proiectului

Acest proiect implementează rețele neuronale pentru super-rezoluție de imagini, permițând scalarea imaginilor cu factori de 2x, 4x, 8x, 16x și 32x, păstrând în același timp calitatea vizuală.

## Arhitectura de Bază (Network2 - upscaling 2x)

### Blocurile Reziduale

```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(p=0.3) 
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x) 
        x = self.conv2(x)
        x = self.bn2(x)
        return x + residual  
```

Fiecare bloc rezidual conține:
- Două straturi convoluționale cu kerneluri 3×3
- Batch normalization pentru stabilitatea antrenării
- Activare ReLU
- Dropout (30%) pentru regularizare
- Conexiune *skip* pentru îmbunătățirea fluxului gradienților

```python
class SRNN(nn.Module):
    def __init__(self):
        super(SRNN, self).__init__()
        self.input_conv = nn.Conv2d(3, 32, kernel_size=9, padding=4)
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(32) for _ in range(5)]
        )
        self.channel_reduction = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.output_conv = nn.Conv2d(16, 3, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_input = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
        x = self.relu(self.input_conv(x))
        x = self.residual_blocks(x)
        x = self.relu(self.channel_reduction(x))
        x = self.upsample(x)
        x = self.output_conv(x)
        return x + x_input  # Conexiune skip globală
```

#### Fluxul rețelei:
- **Extracția Inițială de Caracteristici:** Strat convoluțional 9×9
- **Procesarea Caracteristicilor:** 5 blocuri reziduale
- **Reducerea Canalelor:** Din 32 la 16 canale
- **Upsampling:** Interpolare bicubică + convoluție + ReLU
- **Reconstrucție:** Strat convoluțional final
- **Conexiune Skip Globală:** Upsampling direct + adunare

---

## Procesarea Datelor

Clasa `SRDataset` creează exemple de antrenare prin:
- Luarea unei imagini de înaltă rezoluție
- Extragerea de crop-uri aleatorii
- Aplicarea de augmentări de date:
  - flip-uri
  - rotații
  - modificări de culori
  - blur
- Crearea versiunilor de rezoluție joasă prin **downscaling** cu interpolare bicubică

---

## Procesul de Antrenare

- **Funcția de Pierdere:** `L1 Loss` (Mean Absolute Error)
- **Optimizer:** Adam (`lr=1e-4`, `weight_decay=1e-5`)
- **Scheduler:** Learning rate înjumătățit la fiecare 50 de epoci
- **Metrică de Evaluare:** `PSNR` (Peak Signal-to-Noise Ratio)

---

## Performanță

### Network2 (Upscaling 2x)
- PSNR: **~38.30 dB**

---

### Network4 (Upscaling 4x)
- Bazat pe `Network2` cu:
  - 2 blocuri consecutive de upsampling (2×2x)
  - Rotații extinse (ex: 135°)
  - Crop-uri mai mari
  - Scheduler: step size = 30
- PSNR: **~30.36 dB**

---

### Network8 (Upscaling 8x)
- Îmbunătățiri:
  - 3 blocuri de upsampling (2×2×2x)
  - 12 blocuri reziduale
  - Transformări de perspectivă
  - `lr=2e-4`
  - Scheduler: `gamma=0.3`, `step_size=5`
- PSNR: **~26.06 dB**

---

### Network16 (Upscaling 16x)
- Caracteristici:
  - 4 blocuri de upsampling
  - Augmentări geometrice suplimentare
  - Batch-uri mai mici pentru memorie
- PSNR: **~22.85 dB**

---

### Network32 (Upscaling 32x)
- Caracteristici extreme:
  - 5 blocuri de upsampling
  - Crop-uri foarte mici (8x8)
  - Batch-uri mici (ex: 4)
- PSNR: **~20.38 dB**

---

## Concluzii

Acest proiect demonstrează capacitatea rețelelor neuronale de a efectua upscaling pe imagini cu factori foarte mari. După cum era de așteptat, performanța (măsurată în PSNR) scade odată cu creșterea factorului de upscaling, dar rețelele reușesc să producă rezultate vizual acceptabile chiar și pentru factori extremi precum 32x.
