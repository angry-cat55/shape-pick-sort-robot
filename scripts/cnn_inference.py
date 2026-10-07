"""Colab SmallCNN을 CPU에서 복원하고 같은 RGB crop으로 도형 종류를 예측한다."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn


class SmallCNN(nn.Module):
    def __init__(self, dropout=0.15):
        super().__init__()
        # 학습 노트북과 층·채널 수·순서를 같게 해야 저장된 가중치를 그대로 쓸 수 있다.
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def preprocess(image, mean, std):
    # rgb_crop이 만든64×64 RGB를 채널 우선 텐서로 바꾼다. 추론에서는 밝기 증강을 하지 않는다.
    if (
        not isinstance(image, Image.Image)
        or image.mode != 'RGB'
        or image.size != (64, 64)
    ):
        raise ValueError('CNN 입력은64×64 RGB 사진이어야 합니다')
    tensor = (
        torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1)
        / 255
    )
    return (tensor - torch.tensor(mean)[:, None, None]) / torch.tensor(std)[
        :, None, None
    ]


class CnnClassifier:
    def __init__(self, folder):
        folder = Path(folder)
        self.metadata = json.loads((folder / 'best_params.json').read_text())
        # CPU에 복원하며 weights_only로 가중치·기본 메타데이터만 읽는다.
        checkpoint = torch.load(
            folder / 'best_model.pt', map_location='cpu', weights_only=True
        )
        if checkpoint.get('metadata') != self.metadata or checkpoint.get(
            'params'
        ) != self.metadata.get('params'):
            raise ValueError('가중치와 설정 파일이 같은 학습 결과가 아닙니다')
        m = self.metadata
        if (
            m.get('model') != 'SmallCNN-v1'
            or m.get('classes') != ['cuboid', 'cylinder']
            or m.get('image_size') != [64, 64]
            or m.get('mean') != [0.5] * 3
            or m.get('std') != [0.5] * 3
            or m.get('crop_total_margin_px') != 12
            or m.get('resize') != 'PIL_BILINEAR'
        ):
            raise ValueError('현재 CNN 구조·클래스 순서·전처리와 다른 설정입니다')
        self.model = SmallCNN(m['params']['dropout'])
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.model.eval()  # Dropout을 끄고 학습한 가중치로만 예측한다.
        self.model_sha256 = hashlib.sha256(
            (folder / 'best_model.pt').read_bytes()
        ).hexdigest()
        self.params_sha256 = hashlib.sha256(
            (folder / 'best_params.json').read_bytes()
        ).hexdigest()

    def validate_camera(self, camera):
        # 촬영·crop 조건이 학습 때와 다르면 로봇을 움직이기 전에 중단한다.
        trained = self.metadata.get('camera', {})
        for key in (
            'width',
            'height',
            'eye',
            'target',
            'up',
            'fov_deg',
            'near_m',
            'far_m',
            'roi_xy_m',
            'table_top_m',
            'depth_difference_m',
            'top_band_m',
            'renderer',
            'crop_size',
        ):
            expected, actual = trained.get(key), camera.get(key)
            if isinstance(expected, str):
                same = actual == expected
            else:
                same = (
                    expected is not None
                    and actual is not None
                    and np.allclose(expected, actual, rtol=0, atol=1e-8)
                )
            if not same:
                raise ValueError(f'학습 카메라와 현재 카메라가 다릅니다: {key}')

    def predict(self, image):
        # 수집·학습 때와 같은 정규화를 적용해 모델 입력 형태로 바꾼다.
        tensor = preprocess(image, self.metadata['mean'], self.metadata['std'])
        # 추론에서는 기울기·가중치 갱신 없이 두 클래스의 점수를 확률 형태로 바꾼다.
        with torch.inference_mode():
            logits = self.model(tensor[None])
            if not torch.isfinite(logits).all():
                raise ValueError('CNN 출력이 유한수가 아닙니다')
            probabilities = logits.softmax(dim=1)[0].tolist()
        # 두 점수 중 큰 쪽의 클래스 이름을 골라 좌표 계산 단계에 돌려준다.
        index = int(np.argmax(probabilities))
        return {
            'classification_source': 'cnn',
            'predicted_class': self.metadata['classes'][index],
            'class_score': float(probabilities[index]),
            'class_probabilities': dict(zip(self.metadata['classes'], probabilities)),
        }
