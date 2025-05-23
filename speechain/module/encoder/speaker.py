import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, 1),
            nn.ReLU(),
            nn.Conv1d(bottleneck, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class Res2Block(nn.Module):
    def __init__(self, channels, kernel_size=3, scale=8):
        super().__init__()
        self.scale = scale
        self.width = channels // scale
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(self.width, self.width, kernel_size, padding=kernel_size // 2)
                for _ in range(scale - 1)
            ]
        )

    def forward(self, x):
        splits = torch.split(x, self.width, dim=1)
        out = splits[0]
        outputs = [out]
        for i, (conv, split) in enumerate(zip(self.convs, splits[1:])):
            out = F.relu(out)
            out = conv(out)
            outputs.append(out)
            outputs.append(split)
        return x + torch.cat(outputs[: self.scale], dim=1)


class EncoderClassifier(nn.Module):
    def __init__(self, model_type="ecapa"):
        super().__init__()
        self.model_type = model_type
        self.model = self._create_model()

    def _create_model(self):
        if self.model_type == "ecapa":
            return self._create_ecapa()
        elif self.model_type == "xvector":
            return self._create_xvector()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _create_ecapa(self):
        channels = 512
        model = nn.Sequential(
            nn.Conv1d(80, channels, 7, padding=3),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Sequential(
                Res2Block(channels), SEModule(channels), nn.BatchNorm1d(channels)
            ),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, 192),
        )
        return model

    def _create_xvector(self):
        model = nn.Sequential(
            nn.Conv1d(80, 512, 5, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, 3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, 3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 192),
        )
        return model

    def encode_batch(self, wavs, wav_lens=None):
        self.eval()
        with torch.no_grad():
            x = wavs.transpose(1, 2)
            embeddings = self.model(x)
            return F.normalize(embeddings, p=2, dim=1)

    @classmethod
    def from_hparams(cls, source, savedir=None, run_opts=None):
        """Load pretrained model."""
        model_type = "ecapa" if "ecapa" in source else "xvector"
        model = cls(model_type=model_type)
        
        device = run_opts.get("device", "cpu") if run_opts else "cpu"
        model = model.to(device)

        # Download the model if it doesn't exist
        if savedir and source:
            import urllib.request
            import os
            
            os.makedirs(savedir, exist_ok=True)
            weights_path = os.path.join(savedir, "encoder.pth")
            
            if not os.path.exists(weights_path):
                # Map source names to download URLs
                source_map = {
                    "speechbrain/spkrec-ecapa-voxceleb": "https://github.com/speechbrain/speechbrain/releases/download/v0.5.12/spkrec-ecapa-voxceleb-1b1642a.ckpt",
                    "speechbrain/spkrec-xvect-voxceleb": "https://github.com/speechbrain/speechbrain/releases/download/v0.5.12/spkrec-xvect-voxceleb-6b71f0c.ckpt"
                }
                
                if source in source_map:
                    print(f"Downloading {source} model to {weights_path}")
                    urllib.request.urlretrieve(source_map[source], weights_path)
                else:
                    raise ValueError(f"Unknown source: {source}. Available sources: {list(source_map.keys())}")
            
            # Load pretrained weights
            if os.path.exists(weights_path):
                model.load_state_dict(
                    torch.load(weights_path, map_location=device)
                )
            else:
                raise FileNotFoundError(f"Weights file not found: {weights_path}")

        return model
