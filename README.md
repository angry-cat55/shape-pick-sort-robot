# Shape Pick & Sort Robot

**직접 학습한 CNN과 가상 RGB-D 카메라를 이용해 3D 도형을 인식하고, 로봇암이 실제 접촉으로 집어 분류하는 시뮬레이션 프로젝트입니다.**

ROKEY 부트캠프 개인 프로젝트로, 모델 학습부터 로컬 추론·로봇 제어·결과 검증까지 연결하고 그 흐름을 이해하는 것을 목표로 합니다.

> **현재 상태: 카메라 좌표로 한 종류의 여러 도형을 하나씩 뒤쪽 상자로 운반**
> 다양한 크기·회전의 데이터 수집기와 Colab 랜덤 탐색 노트북을 준비했습니다. 본 학습과 로컬 CNN 연결은 다음 순서입니다.
> 하나를 옮기고 팔을 대기 자세로 돌린 뒤 다시 촬영합니다. 도형 종류는 명령으로 지정하며, 혼합 도형의 CNN 분류·ROS2·실패 자동 재시도는 후속 단계입니다.

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

**같은 종류를 하나씩 운반하려면** `--task sort`를 추가합니다. GUI는 작업이 끝나면 종료됩니다.

```bash
.venv/bin/python scripts/multi_object_scene.py --mode gui --task sort --cuboids 5 --cylinders 0 --seed 0
.venv/bin/python scripts/multi_object_scene.py --mode gui --task sort --cuboids 0 --cylinders 5 --seed 0
```

CNN 연결 전에는 혼합 종류 운반을 거부합니다. 현재는 기본 크기의 분리된 직육면체·수직 원기둥만 대상으로 합니다. 일부 잘린 윗면은 보류하지만 모든 가림을 알아내는 기능은 아닙니다. 잡을 대상이 없는데 물체가 남아 있으면 완료 대신 실패로 기록합니다.

## 파일 구조와 역할

파일 수가 늘어난 대부분의 이유는 실행 코드에 더해 테스트·설명·실험 기록을 남겼기 때문입니다. 현재 실행 코드의 중심은 `scripts/`의 다섯 파일이며, 학습 코드는 `notebooks/`에 있습니다.

```text
shape-pick-sort-robot/
├── README.md                   프로젝트 소개·실행 명령·현재 상태
├── requirements-sim.txt        시뮬레이션에 설치할 Python 패키지 목록
├── .gitignore                 Git에 올리지 않을 파일·폴더 설정
├── AGENTS.md                  AI 작업 지침 [로컬 전용]
├── scripts/                   직접 실행하거나 함께 사용하는 코드
│   ├── pybullet_first_run.py   설치 후 로봇·중력·바닥을 확인하는 첫 실험
│   ├── contact_grasp_probe.py  한 물체의 접촉 집기·이동·성공/실패 검증
│   ├── rgbd_camera.py          RGB-D 촬영·영역 분리·좌표/방향/폭 계산
│   ├── multi_object_scene.py   최대5개 배치·카메라 표시·하나씩 운반·재촬영
│   └── collect_cnn_data.py     크기·회전 랜덤 RGB 수집·장면별 데이터 분할
├── tests/                     코드를 실행해서 동작을 확인하는 자동 테스트
│   ├── test_pybullet_first_run.py    첫 장면·입력·종료 확인
│   ├── test_contact_grasp_probe.py   집기·낙하·상자 도착 대조 실험
│   ├── test_rgbd_camera.py           깊이 좌표 변환·도형 기하 확인
│   ├── test_multi_object_scene.py   여러 물체 배치·순차 운반·재관측 확인
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
├── checkpoints/               Colab에서 가져올 학습 가중치 [Git 제외]
├── outputs/                   실행하면서 만든 영상·설정·결과·접촉 로그
├── .venv/                     이 프로젝트의 Python 실행 환경과 설치 패키지
├── .superpowers/              AI 작업 도구의 임시 진행 기록
└── .git/                      Git이 관리하는 커밋·브랜치 기록
```

`multi_object_scene.py`가 전체 작업 순서를 진행하고, `rgbd_camera.py`에서 좌표를 받아 `contact_grasp_probe.py`의 제어·검증을 재사용합니다. `pybullet_first_run.py`는 초기 설치 확인용으로 남겨둡니다.

`outputs/`, `.venv/`, `.superpowers/`와 로컬 전용 문서는 Git 업로드 대상에서 제외합니다. 여러 물체 설계·계획과 순차 운반 계획 세 문서도 이전 제외 요청에 따라 현재 로컬에만 남겨뒀습니다. CNN 데이터 수집 코드와 Colab 학습 노트북을 추가했습니다. CNN 로컬 추론 연결과 ROS2 패키지는 후속 단계입니다.

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
- [ ] 데이터 자동 생성·CNN 학습 및 평가
- [ ] 모델 저장·로컬 추론 연결
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

직육면체 가로·세로 3~6.4cm, 수직 원기둥 지름 3~6.4cm, 두 종류 높이 3~7cm를 랜덤 생성합니다. Yaw도 −180~180도로 바꿉니다. 현재 여러 물체 장면의 65도 카메라와 64×64 RGB crop을 그대로 사용합니다. 새 크기의 운반 성공을 모두 보장하는 범위는 아닙니다.

```bash
.venv/bin/python -m pip install -r requirements-sim.txt
.venv/bin/python scripts/collect_cnn_data.py --scenes 2000 --seed 2026
```

2,000개 장면에서 예상 약 6,000장의 사진을 수집하며 실제 사진 수는 개수·재생성 결과에 따라 달라집니다. 장면 기준으로 학습/검증/테스트를 70/15/15로 나눕니다. 같은 장면의 물체들은 같은 분할에 들어가고, 각 분할은 두 클래스를 모두 포함합니다. `config.json`에 실제 클래스별 사진 수, `manifest.jsonl`에 크기·Yaw·장면·라벨을 저장합니다. 같은 출력 폴더는 덮어쓰지 않습니다. 실패하거나 중단한 수집은 incomplete로 표시되며 학습에 사용하지 않습니다.

결과는 Git에서 제외한 `data/shape_cnn_<실행시간>/`와 같은 이름의 zip에 저장합니다. zip을 개인 Google Drive에 업로드하고 [Colab 노트북](notebooks/shape_cnn_colab.ipynb)을 열어 경로를 지정합니다. tqdm 수집·학습 진행바가 표시됩니다. crop은 배경을 포함하므로 여백에 이웃 물체 일부가 보일 수 있습니다. 라벨은 깊이로 찾은 목표 영역의 물체를 가리킵니다.

노트북은 기본 12회 랜덤 탐색(최대20epoch·조기종료4epoch)으로 학습률, Adam/AdamW/SGD, 교차엔트로피/스무딩 교차엔트로피, 배치 크기·weight decay·dropout을 비교합니다. 검증 macro F1로 최선 설정을 고르고 동률이면 공통 NLL이 낮은 모델을 고릅니다. 테스트는 모델 선택 뒤에만 평가합니다.

정확도·F1·precision·recall·ROC-AUC·손실 곡선, 실험 비교, 혼동행렬·ROC, 오분류 사진과 크기별 지표를 저장합니다. `best_model.pt`와 `best_params.json`에는 가중치·클래스 순서·전처리·카메라·데이터 해시·버전도 포함됩니다. 노트북 코드는 Git 관리하고 데이터·가중치·실행 결과는 개인 Drive 또는 로컬 제외 폴더에 보관합니다. 처음 Colab 실행은 2회·3epoch로 경로와 동작을 확인한 뒤 탐색량을 늘리세요.

7cm 높이는 독립 단일 물체 실험에서 확인했지만 모든 크기·배치의 운반 성공률은 아닙니다. 정사각형 윗면의 방향/폭 추정과 기존 기본 크기 전용 가림 검사는 다양한 크기 집기 연결 단계에서 보완해야 합니다. 이번 단계는 학습 준비이며, 혼합 종류 자동 운반·CNN 로컬 추론·ROS2는 아직 연결하지 않았습니다.

이번에 생성한 로컬 데이터는 `data/shape_cnn_v1.zip`(약11MB)입니다. 2,000개 장면·사진6,065장이고, 검증한 실제 분할은 아래와 같습니다. 장면 겹침과 분할 간 동일 이미지 중복은 없었습니다.

| 분할 | 직육면체 | 원기둥 | 총 사진 |
|---|---:|---:|---:|
| 학습 | 2,157 | 2,113 | 4,270 |
| 검증 | 449 | 450 | 899 |
| 테스트 | 435 | 461 | 896 |

자동 테스트36개 통과, 실제 작은 데이터에서 CPU 축소 학습·평가·그래프·가중치 저장/복원 확인을 완료했습니다. Colab GPU 본 학습 성능은 사용자가 학습한 뒤 평가합니다.

파라미터별 검증 F1 비교 그림에는 선택된 실험을 빨간 별로 표시합니다. 옵티마이저·손실 설정·배치 크기·dropout·학습률·weight decay를 비교하고, 선택 이유는 `selection_reason.txt`, 전체 설정·점수는 `parameter_evidence.csv`에 저장합니다. 이미 학습한 결과에서도 해당 비교 셀만 실행할 수 있습니다. 여러 설정을 동시에 바꾼 결과라 한 파라미터만의 효과로 단정하지 않습니다.
