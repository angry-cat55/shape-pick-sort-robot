# Shape Pick & Sort Robot

**직접 학습한 CNN과 가상 RGB-D 카메라를 이용해 3D 도형을 인식하고, 로봇암이 실제 접촉으로 집어 분류하는 시뮬레이션 프로젝트입니다.**

ROKEY 부트캠프 개인 프로젝트로, 모델 학습부터 로컬 추론·로봇 제어·결과 검증까지 연결하고 그 흐름을 이해하는 것을 목표로 합니다.

> **현재 상태: CNN으로 두 종류를 분류하고, 카메라 좌표로 하나씩 종류별 상자에 운반**
> Colab에서 학습한 모델을 CPU 추론에 연결했습니다. 기본 크기 혼합 장면6개에서30개 모두 올바르게 분류·운반했고, 하나를 옮길 때마다 다시 촬영합니다.
> 랜덤 크기 생성과 영상 기반 가림 보류를 추가했습니다. 실패 자동 재시도·ROS2 연결은 후속 단계입니다.

## 첫 장면 실행

프로젝트 루트에서 실행합니다. GUI는 약 60초 뒤 자동 종료되며, Ctrl+C로 먼저 중단할 수 있습니다.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-sim.txt
.venv/bin/python scripts/pybullet_first_run.py --mode gui --steps 14400
```

화면 없이 검증하려면 `--mode direct --steps 480`을 사용합니다. 설치와 실행 흐름은 [PyBullet 시작 안내](docs/PYBULLET_START_KO.md)에 설명했습니다.

고정 직육면체 집기를 화면으로 보려면 다음을 실행합니다. 약 14초 후 종료됩니다.

```bash
.venv/bin/python scripts/contact_grasp_probe.py --mode gui --scenario normal
```

같은 고정 조건에서 정상 집기 10회와 실패 대조군을 재현하려면 `--scenario all --trials 10`을 사용합니다. 이 결과는 물체의 다양한 위치·크기·방향에 대한 성공률이 아닙니다.

파란 상자 운반을 보려면 같은 파일에서 `--mode gui --scenario transport`를 사용합니다. 정상 운반10회와 실패 대조군은 `--scenario all_transport --trials 10`으로 실행합니다.

카메라에서 계산한 좌표로 집고 운반하려면 다음을 실행합니다. 기존 명령은 기본값인 `oracle`(고정 정답 좌표) 실험으로 유지합니다.

```bash
.venv/bin/python scripts/contact_grasp_probe.py --mode gui --pose-source camera --scenario transport
```

`--block-x 0.48 --block-y -0.025 --block-yaw -0.35`를 추가하면 물체 초기 위치·회전을 바꿀 수 있습니다. 값은 미터·라디안 단위이며, 현재 검증 범위는 시작 안내에서 확인합니다.

수직 원기둥을 집어 운반하려면 같은 프로그램에 도형 종류를 지정합니다.

```bash
.venv/bin/python scripts/contact_grasp_probe.py --mode gui --pose-source camera --object-shape cylinder --scenario transport
```

직육면체 기본 크기는 6×4×6cm, 원기둥 기본 지름은 5cm, 높이는 6cm입니다. 상자 벽은 8cm입니다. `--cylinder-diameter 0.025 --cylinder-height 0.03`으로 작은 원기둥을 검사할 수 있습니다. 정상10회와 대조군은 GUI 옵션 없이 `--scenario all_transport --trials 10`으로 실행합니다.

## 여러 도형을 배치하는 새 장면

로봇 앞의 청록색 구역에 선택한 개수의 도형을 떨어뜨리고, 뒤쪽에는 두 종류의 상자를 배치합니다. 합계는 1~5개이며, 작은 카메라와 노란 촬영 범위 테두리도 표시합니다.

```bash
.venv/bin/python scripts/multi_object_scene.py --mode gui --cuboids 3 --cylinders 2 --seed 0 --duration 30
```

`--cuboids`는 직육면체 개수, `--cylinders`는 원기둥 개수, `--seed`는 랜덤 배치를 재현하는 번호입니다. `--mode direct`는 화면 없이 검증합니다. 기본 실행은 장면 확인이며 로봇이 운반하지 않습니다.

**같은 종류를 하나씩 운반하려면** `--task sort`를 추가합니다. GUI는 작업이 끝나면 종료됩니다. 터미널의 동작 단계·결과·입력 오류는 한글로 표시하며, 저장 JSON의 상태 식별자는 기존 영문을 유지합니다.

```bash
.venv/bin/python scripts/multi_object_scene.py --mode gui --task sort --cuboids 5 --cylinders 0 --seed 0
.venv/bin/python scripts/multi_object_scene.py --mode gui --task sort --cuboids 0 --cylinders 5 --seed 0
```

**두 종류를 CNN으로 분류해서 운반하려면** 아래처럼 실행합니다. 추론 패키지는 별도로 설치하며, Colab에서 가져온 `best_model.pt`와 `best_params.json`이 `checkpoints/shape_cnn_v1/`에 있어야 합니다.

```bash
.venv/bin/python -m pip install -r requirements-inference.txt
.venv/bin/python scripts/multi_object_scene.py --mode gui --task sort --classification-source cnn --cuboids 3 --cylinders 2 --seed 0
```

깊이로 전체 영역 찾기 → 윗면 후보 점이 많은 순서로 정렬 → 앞 후보의 RGB만64×64로 잘라 CNN 분류·좌표 검사 → 통과하면 운반 → 다시 촬영하는 순서입니다. 후보가 탈락하면 같은 영상의 다음 후보를 확인하고, 하나가 통과하면 나머지는 분류하지 않습니다. 기본 크기5개 혼합 장면6개에서 같은 조건의 기존15회 분류가5회로 줄었고,30개 모두 올바른 상자에 도착했습니다. 후보 탈락이 있으면 분류 횟수는 더 늘 수 있습니다. 상자마다 다음 빈 자리를 따로 셉니다. 기본값인 수동 모드는 기존 한 종류 실험을 유지합니다.

`--min-class-score` 기본값은0.8이며 그보다 낮은 예측은 보류합니다. 이 점수가 '실제로80% 맞는다'는 뜻은 아닙니다. 분류 점수가 높아도 잘린 윗면·폭·높이 등 기존 검사를 통과해야 집습니다. 학습과 다른 카메라 설정이나 서로 맞지 않는 모델·설정 파일은 시작 전에 거부합니다.

 기본 모드는 기존 크기와15% 윗면 검사를 유지합니다. 랜덤 크기는 아래 옵션으로 선택합니다. 일부 가림은 보류하지만 모든 가림을 알아내는 기능은 아닙니다. 잡을 대상이 없는데 물체가 남아 있으면 완료 대신 실패로 기록합니다.

## 랜덤 크기 도형을 분류하고 운반하기

`--size-mode random`을 추가하면 **각 물체마다** 직육면체 가로·세로 또는 원기둥 지름을 3 ~ 6.4cm, 높이를 3 ~ 7cm로 뽑습니다. 위치와 Yaw도 랜덤이며, 같은seed로 같은 시도를 재현할 수 있습니다. 합계는 최대5개입니다.

```bash
.venv/bin/python scripts/multi_object_scene.py --mode gui --task sort --classification-source cnn --size-mode random --cuboids 3 --cylinders 2 --seed 0
```

창 없이 검사하려면 `--mode direct`로 바꿉니다. `--task sort --classification-source cnn`을 빼면 생성된 장면만 볼 수 있습니다. 모델·설정 파일은 `checkpoints/shape_cnn_v1/`에 있어야 합니다.

범위를 좁히려면 `--min-width-cm 3 --max-width-cm 5 --min-height-cm 3 --max-height-cm 6`처럼 지정합니다. 학습 데이터 범위인 폭 3 ~ 6.4cm·높이 3 ~ 7cm 밖은 거부합니다. 범위는 허용 조건이며, 개별 생성 치수를 파지 좌표 계산에 넘기지 않습니다.

**카메라 검사:** 추정한 폭·높이가 허용 범위인지 확인하고, 옆면 점이 윗면 테두리 밖으로 퍼진 붙은 영역을 보류합니다. 더 높은 물체가 후보의 가능한 윗면 범위를 가릴 수 있으면 다른 물체를 먼저 옮긴 뒤 다시 촬영합니다. 보류된 후보가 있으면 CNN 호출은 물체 수보다 많아질 수 있습니다.

원기둥 지름은 상면의 여러 테두리 점에 원을 맞춰 구합니다. 한두 끝 픽셀의 오차로 최대 지름이 크게 계산되는 문제를 줄이면서 **기존 집기 폭 상한6.5cm를 유지**합니다. 원과 맞지 않는 경계는 보류합니다.

가림 검사는 현재 카메라·수직 도형·분리된 배치에 한정된 보수적 검사입니다. 보이지 않는 모양을 완전히 복원하거나 모든 겹침·기울어짐을 처리하는 기능은 아닙니다. 초기 배치가 간격·정착 조건을 만족하지 않으면 기록하고 다음seed로 다시 생성하며, 운반 중 순간이동으로 고치지 않습니다.

**이번 검증:** 랜덤 크기12회와 최소·최대 크기8회, 합계20회 실행에서100개 물체의 분류·집기·상자 도착·최종 정착을 확인했습니다. 일부 초기 배치는 다음seed로 재생성되어 같은 장면이 반복되었으며, 이 결과가 모든 랜덤 배치의 성공률을 뜻하지는 않습니다. 랜덤 크기 운반은 접촉 계산을100회에서200회로 늘려 원기둥의 바닥 접촉 흔들림을 줄였습니다. 마지막 확인은 최대5초 기다리며, 모든 물체가 기존 속도·바닥 접촉 기준을0.5초 연속 만족해야 통과합니다.

전체 자동 테스트는 아래 명령으로 실행합니다. 이번 환경에서는52개가 통과했으며, 개인 모델 파일이 없는 환경에서는 모델을 쓰는 일부 테스트가 건너뛰어집니다.

```bash
.venv/bin/python -m unittest discover -s tests
```

## 파일 구조와 역할

파일 수가 늘어난 대부분의 이유는 실행 코드에 더해 테스트·설명·실험 기록을 남겼기 때문입니다. 현재 실행 코드의 중심은 `scripts/`에 있으며, 학습 코드는 `notebooks/`에 있습니다.

```text
shape-pick-sort-robot/
├── README.md                   프로젝트 소개·실행 명령·현재 상태
├── requirements-sim.txt        시뮬레이션에 설치할 Python 패키지 목록
├── requirements-inference.txt  로컬 CNN 추론에 필요한 CPU PyTorch
├── .gitignore                 Git에 올리지 않을 파일·폴더 설정
├── AGENTS.md                  AI 작업 지침 [로컬 전용]
├── scripts/                   직접 실행하거나 함께 사용하는 코드
│   ├── pybullet_first_run.py   설치 후 로봇·중력·바닥을 확인하는 첫 실험
│   ├── contact_grasp_probe.py  한 물체의 접촉 집기·이동·성공/실패 검증
│   ├── rgbd_camera.py          RGB-D 촬영·영역 분리·좌표/방향/폭 계산
│   ├── multi_object_scene.py   최대5개 배치·카메라 표시·하나씩 운반·재촬영
│   ├── terminal_ko.py          터미널 단계·실패 원인·도움말의 공통 한글 표시
│   ├── cnn_inference.py        Colab 모델 복원·입력 정규화·도형 종류 예측
│   └── collect_cnn_data.py     크기·회전 랜덤 RGB 수집·장면별 데이터 분할
├── tests/                     코드를 실행해서 동작을 확인하는 자동 테스트
│   ├── test_pybullet_first_run.py    첫 장면·입력·종료 확인
│   ├── test_contact_grasp_probe.py   집기·낙하·상자 도착 대조 실험
│   ├── test_rgbd_camera.py           깊이 좌표 변환·도형 기하 확인
│   ├── test_multi_object_scene.py   여러 물체 배치·순차 운반·재관측 확인
│   ├── test_random_size_sort.py   랜덤 치수·크기 경계·가림/붙은 영역 보류 확인
│   ├── test_cnn_inference.py       학습·추론 일치·불확실한 예측·혼합 운반 확인
│   └── test_cnn_data.py             크기·라벨·데이터 분할 확인
├── notebooks/                 Colab에서 실행하는 학습 코드
│   └── shape_cnn_colab.ipynb   랜덤 탐색·모델 저장·성능 시각화
├── docs/                      사람이 읽는 설명과 프로젝트 기록
│   ├── PYBULLET_START_KO.md    설치·단계별 동작·실행법·검증 범위
│   ├── DECISIONS_KO.md         선택한 방식·다른 후보·선택 이유
│   ├── FOLLOW_UP_KO.md         강화학습 등 나중에 할 후보와 보류 시점
│   ├── experiments.csv        실험별 설정·결과·원본 로그 위치 표
│   ├── HANDOFF_FULL_KO.md      Windows 대화와 프로젝트 배경 [로컬 전용]
│   ├── PROGRESS.md             최신 진행 상태·인계 기록 [로컬 전용]
│   ├── images/cnn/            README에 넣은 대표 CNN 결과 그래프4장
│   └── superpowers/           개발 전에 작성한 설계와 단계별 작업 계획
│       ├── specs/             무엇을 만들지 정한 설계 명세
│       │   ├── 2026-10-04-shape-sort-design.md         최초 전체 설계
│       │   └── 2026-10-04-multi-object-scene-design.md 최대5개·뒤 상자 설계
│       └── plans/             어떤 순서로 구현·검증할지 정한 계획
│           ├── 2026-10-04-pybullet-first-run.md 첫 장면
│           ├── 2026-10-04-fixed-contact-grasp.md 고정 물체 집기
│           ├── 2026-10-04-bin-transport.md       상자 운반
│           ├── 2026-10-04-rgbd-camera.md         카메라 좌표 추정
│           ├── 2026-10-04-cylinder-grasp.md      원기둥 집기
│           ├── 2026-10-04-multi-object-scene.md  여러 물체 구도
│           └── 2026-10-05-sequential-pick.md     하나씩 운반·재촬영
├── data/                      CNN 사진·라벨·데이터 zip [Git 제외]
├── checkpoints/               Colab 최종 모델·설정 [Git 제외]
├── outputs/                   실행하면서 만든 영상·설정·결과·접촉 로그
├── .venv/                     이 프로젝트의 Python 실행 환경과 설치 패키지
├── .superpowers/              AI 작업 도구의 임시 진행 기록
└── .git/                      Git이 관리하는 커밋·브랜치 기록
```

`multi_object_scene.py`가 전체 작업 순서를 진행하고, `rgbd_camera.py`에서 좌표를 받아 `contact_grasp_probe.py`의 제어·검증을 재사용합니다. `pybullet_first_run.py`는 초기 설치 확인용으로 남겨둡니다.

`outputs/`, `.venv/`, `.superpowers/`와 로컬 전용 문서는 Git 업로드 대상에서 제외합니다. 여러 물체 설계·계획과 순차 운반 계획 세 문서도 이전 제외 요청에 따라 현재 로컬에만 남겨뒀습니다. CNN 데이터 수집 코드와 Colab 학습 노트북을 추가했습니다. CNN 로컬 추론과 혼합 운반을 연결했으며 ROS2 패키지는 후속 단계입니다.

## 프로젝트 목표

- 시뮬레이터에서 생성한 데이터로 사전학습 가중치 없이 작은 CNN을 학습합니다.
- RGB 영상으로 도형 종류를 분류하고 깊이 영상으로 파지에 필요한 위치·방향·폭을 추정합니다.
- 역기구학과 그리퍼 제어로 물체를 집어 종류별 상자로 옮깁니다.
- 집기 실패와 운반 중 낙하를 확인하고 재관찰·제한된 재시도를 수행합니다.
- 분류 성능과 로봇 작업 성공률을 구분해 평가합니다.

## 동작 흐름

```text
도형 생성 → RGB-D 관찰 → 물체 영역 추출
                           ├─ CNN: 종류 분류
                           └─ 기하학: 위치·방향·폭 추정
                                      ↓
                           파지 자세 계산 · 역기구학
                                      ↓
                             접근 · 그리퍼 닫기 · 들기
                                      ↓
                                집기 결과 검증
                           ┌──────────┴──────────┐
                        성공                    실패
                  상자 운반 · 내려놓기     재관찰 · 제한된 재시도
                           ↓
                       도착 결과 검증
```

## 첫 버전 범위

| 항목 | 계획 |
|---|---|
| 시뮬레이션 | PyBullet |
| 로봇 | Franka Panda와 두 손가락 그리퍼 후보 |
| 물체 | 직육면체·수직 원기둥 합계 최대 5개, 하나씩 운반 |
| 관찰 | 고정 가상 RGB-D 카메라 |
| 파지 | 위에서 접근하는 기하학·제어 기반 파지 |
| 분류 모델 | 사전학습 없이 직접 학습하는 작은 CNN |
| 명령·상태 | 터미널 기반 스폰·시작·정지·상태 표시 |
| ROS2 | 필수 범위: 기본 흐름 검증 후 영상·인식·작업 상태 통신 연동 |

첫 버전에서는 강화학습 기반 파지, 여러 물체의 겹침·쌓임, 임의 형상, 실제 로봇 제어를 다루지 않습니다. 로봇·물체·카메라의 상세 설정은 실험 결과에 따라 조정합니다.

## 검증 원칙

- **물리적 집기:** 물체를 그리퍼에 강제로 붙여 성공을 대신하지 않습니다.
- **정보 경계:** 최종 파지 위치는 카메라로 추정합니다. 시뮬레이터 정답은 라벨·오차 평가·결과 검증에 사용합니다.
- **역할 구분:** CNN은 종류를 분류하고, 기하학·역기구학·제어 로직이 파지와 이동을 담당합니다.
- **평가 분리:** 분류 오류, 집기 실패, 운반 낙하, 잘못된 상자 도착을 별도로 기록합니다.
- **실험 우선:** 고정 직육면체의 접촉 파지가 검증되기 전에는 데이터 대량 수집이나 전체 통합을 진행하지 않습니다.

## 개발 순서

- [x] Ubuntu 기본 환경 확인 및 프로젝트 전용 환경 구성
- [x] PyBullet 첫 장면의 중력·바닥 충돌·GUI 검증
- [x] 고정 직육면체의 접촉 기반 집기·실패 판정
- [x] 고정 물체의 상자 운반·내려놓기 검증
- [x] RGB-D 기반 위치·방향 추정 연결
- [x] 한 종류 여러 물체의 순차 운반과 물체마다 재촬영
- [x] 데이터 자동 생성·CNN 학습 및 평가
- [x] 모델 저장·로컬 추론 연결
- [ ] 전체 작업과 실패 대응 통합
- [ ] ROS2 연동 및 시연·발표 정리

## 평가 계획

**모델:** 학습·검증 loss, 분류 정확도, 클래스별 혼동행렬, 조건별 오분류, 설정별 실험 비교.

**시스템:** 첫 집기 성공률, 재시도 포함 성공률, 운반 낙하, 올바른 상자 도착률. 필요하면 위치 추정 오차와 처리 시간도 측정합니다.

성능 수치와 시연 영상은 실제 실험 후 추가합니다.

## 기술 구성

Python · PyBullet · PyTorch · NumPy · OpenCV · ROS2

시뮬레이션·추론은 Ubuntu, 모델 학습은 Google Colab을 사용할 계획입니다. 첫 실행은 Ubuntu 24.04.5 / Python 3.12.3 / PyBullet 3.2.7에서 검증했습니다. ROS2 Jazzy의 명령과 Python import를 확인했으며, 프로젝트 통신 연결은 후속 단계입니다.

## 문서

- [설계 명세](docs/superpowers/specs/2026-10-04-shape-sort-design.md)
- [PyBullet 시작 안내](docs/PYBULLET_START_KO.md)
- [의사결정 기록](docs/DECISIONS_KO.md)
- [실험 기록 양식](docs/experiments.csv)

작은 검증 단계별로 구현하고 결과를 명세와 실험 기록에 반영하는 가벼운 SDD 방식으로 진행합니다.

## CNN 데이터 수집과 Colab 학습

**직육면체와 원기둥을 구분하는 CNN 학습을 완료했습니다.** PyBullet에서 사진을 만들고, Colab의 NVIDIA L4 GPU로 여러 설정을 학습한 뒤 검증 결과로 모델을 골랐습니다. 최종 테스트에서는896장 모두 올바르게 분류했습니다.

### 1. 사진 수집과 데이터 분할

직육면체 가로·세로와 원기둥 지름은 3 ~ 6.4cm, 높이는 3 ~ 7cm로 바꾸고 Yaw도 −180 ~ 180도로 랜덤하게 돌려 촬영했습니다. 로봇 실행과 같은65도 대각선 카메라 위치에서 깊이로 물체 영역을 찾고, 그 영역을 **64×64 RGB 사진**으로 잘랐습니다. CNN에는 깊이가 아닌 RGB만 넣습니다.

2,000개 장면에서 사진6,065장을 모았습니다. 같은 장면의 사진이 학습과 평가에 섞이지 않도록 **장면 기준70:15:15**로 나눴고, 장면 겹침과 분할 간 동일 이미지 중복이 없음을 확인했습니다.

| 분할 | 직육면체 | 원기둥 | 총 사진 |
|---|---:|---:|---:|
| 학습 | 2,157 | 2,113 | 4,270 |
| 검증 | 449 | 450 | 899 |
| 테스트 | 435 | 461 | 896 |

![학습·검증·테스트 사진 수와 클래스별 구성](docs/images/cnn/dataset_counts.png)

학습 사진에만 밝기·대비를 0.9 ~ 1.1배로 바꾸는 증강을 적용했습니다. 픽셀은255로 나눠 0 ~ 1로 바꾼 뒤, `(값 - 0.5) / 0.5`로 −1 ~ 1 범위에 맞췄습니다. 검증·테스트와 이후 로컬 추론도 같은 정규화를 사용합니다. crop 여백에 이웃 물체 일부가 보일 수 있으며, 라벨은 깊이로 찾은 목표 영역의 물체를 가리킵니다.

### 2. 작은 CNN으로 여러 설정 학습하기

사전학습 가중치 없이 `SmallCNN`을 처음부터 학습했습니다. **Conv → ReLU → MaxPool을3회** 반복해 특징을 찾고, AdaptiveAvgPool과 분류층을 거쳐 두 클래스의 점수를 출력합니다. Dropout은 학습 중 분류층의 일부 출력을 잠시 꺼서 특정 특징에만 의존하는 것을 줄입니다.

이번 Colab 실행에서는 **30가지 설정 조합**을 만들고, 조합마다 최대30epoch 학습했습니다. 학습률·Adam/AdamW/SGD·교차엔트로피/라벨 스무딩·배치 크기·weight decay·dropout을 바꿨습니다. 각epoch마다 검증해 가장 좋은 가중치를 저장하고, 검증 F1과 동률일 때의 NLL을 기준으로6epoch 동안 개선이 없으면 해당 조합을 멈췄습니다. 별도로 검증 F1의 정체를 확인하는 설정(`patience=2`)으로 학습률을 절반으로 줄였습니다.

![선택 모델의 학습·검증 손실, 정확도와 검증 지표](docs/images/cnn/best_learning_curves.png)

**읽는 순서:** 왼쪽에서 손실이 줄어드는지, 가운데에서 학습·검증 정확도가 함께 올라가는지 봅니다. 선택된 모델은 초반에 빠르게 도형을 구분했고, 오른쪽의 F1·precision·recall·ROC-AUC도1.0에 도달했습니다. 지표들이 같은 값이라 선이 겹쳐 보입니다.

### 3. 어떤 설정을 최종 모델로 골랐나?

검증 **macro F1이 가장 높은 모델**을 먼저 고릅니다. 30개 중23개가 F1=1.0으로 같아서, **공통 NLL이 가장 낮은27번 실험**을 선택했습니다. NLL은 정답 클래스에 준 확률이 높을수록 작아지는 값입니다. 라벨 스무딩 유무와 관계없이 같은 기준으로 비교했습니다. 테스트 사진은 이 선택에 사용하지 않았습니다.

| 항목 | 최종 설정 |
|---|---|
| 옵티마이저·손실 | AdamW · CrossEntropyLoss |
| 초기 학습률 | 약0.007415 |
| 배치 크기 | 128 |
| Weight decay·Dropout | 약0.00001675 · 0.15 |
| 채택한 가중치 | 27번 실험의20epoch 시점 |
| 검증 F1·NLL | 1.0 · 약5.70×10⁻⁹ |

![설정별 검증 F1 비교와 선택된27번 조합의 빨간 별](docs/images/cnn/parameter_comparison.png)

**빨간 별은 선택한 조합의 설정입니다.** 대부분 F1=1.0이라 이 그림만으로27번이 더 좋다고 구별하기는 어렵습니다. 예를 들어8번도 F1=1.0이지만 NLL은 약8.23×10⁻⁸로27번보다 높았습니다. 여러 설정을 동시에 바꾼 탐색이므로 “AdamW 하나 때문에 좋아졌다”처럼 특정 설정의 효과로 단정하지는 않습니다.

### 4. 선택 후 처음 보는 테스트 사진으로 평가

![테스트 혼동행렬, ROC 곡선과 최종 성능](docs/images/cnn/test_scores_confusion_roc.png)

혼동행렬에서 직육면체435장과 원기둥461장이 모두 대각선에 있고, 반대 클래스로 틀린 칸은0입니다. 테스트 정확도·macro precision·recall·F1·ROC-AUC는 모두1.0이고 오분류는 없었습니다. **현재 시뮬레이션 데이터에서의 분류 결과**이며, 실제 환경 전체의 정확도나 로봇의 집기 성공률을 뜻하지 않습니다.

### 5. 로컬 추론과 로봇 작업 연결

- `checkpoints/shape_cnn_v1/best_model.pt`: 학습한 가중치와 모델 복원 정보.
- `checkpoints/shape_cnn_v1/best_params.json`: 클래스 순서·정규화·카메라·선택 설정·데이터 해시.

두 파일을 같은 Colab 실행(`20261004T212550_059290`)에서 복사했습니다. `cnn_inference.py`는 학습 때와 같은 모델 구조·정규화로 CPU에서 예측합니다. 같은 테스트896장으로 저장된 Colab 출력과 비교했으며, 예측 종류는 모두 같고 두 클래스 점수의 최대 차이는 약3.57×10⁻⁸이었습니다. 모델을 다시 학습하거나 테스트 결과로 설정을 바꾸지는 않았습니다.

이 모델을 `multi_object_scene.py`에 연결해, 기본 크기 직육면체3+원기둥2 및 직육면체2+원기둥3을 각각 seed 0 ~ 2에서 검증했습니다. **6장면·30개 모두 올바른 분류·들림·종류별 상자 도착·최종 정착을 확인**했습니다. 분류 정답 여부와 실제 상자 도착 여부는 별도 기록합니다. 이는 선택한 기본 크기 조건의 결과이며, 임의 크기나 모든 가림의 성공률은 아닙니다.

원본 관측·crop·클래스 점수·접촉 로그는 `outputs/cnn-local-validation/`에 로컬로 보관합니다. 현재는 실패를 구분하고 중단하며 자동 재시도와 ROS2는 다음 작업입니다.

사진·zip·가중치·전체 실행 결과와 개인 학습 정리는 Git에서 제외합니다. [Colab 학습 코드](notebooks/shape_cnn_colab.ipynb)와 README에 사용하는 대표 그래프4장만 저장소에 남깁니다. 원본 실험 기록은 개인 Drive와 로컬 `outputs/colab-result-review/`에 보관합니다.

데이터 수집을 다시 실행하려면 아래 명령을 사용합니다. 실행마다 새 출력 폴더를 만들고, 사진·라벨·분할을 저장한 zip을 개인 Drive로 옮겨 Colab에서 학습합니다. 노트북 기본값은12회·최대20epoch·조기종료4epoch이며, **이번 본 학습은30회·30epoch·6epoch로 늘린 결과**입니다.

```bash
.venv/bin/python -m pip install -r requirements-sim.txt
.venv/bin/python scripts/collect_cnn_data.py --scenes 2000 --seed 2026
```
