# Shape Pick & Sort Robot

**직접 학습한 CNN과 가상 RGB-D 카메라를 이용해 3D 도형을 인식하고, 로봇암이 실제 접촉으로 집어 분류하는 시뮬레이션 프로젝트입니다.**

ROKEY 부트캠프 개인 프로젝트로, 모델 학습부터 로컬 추론·로봇 제어·결과 검증까지 연결하고 그 흐름을 이해하는 것을 목표로 합니다.

> **현재 상태: CNN으로 두 종류를 분류하고, 카메라 좌표로 하나씩 종류별 상자에 운반**
> Colab에서 학습한 모델을 CPU 추론에 연결했습니다. 기본 크기 혼합 장면6개에서30개 모두 올바르게 분류·운반했고, 하나를 옮길 때마다 다시 촬영합니다.
> 랜덤 크기 생성·영상 기반 가림 보류와 집기 실패·낙하 후 제한된 재시도를 추가했습니다. ROS2 노드 세 개로 관측·인식 결과·운반 명령을 주고받는 실행도 추가했습니다.

## 첫 장면 실행

프로젝트 루트에서 실행합니다. 첫 장면은 약 20초 동안 표시되며, Ctrl+C로 먼저 중단할 수 있습니다.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-sim.txt
.venv/bin/python scripts/multi_object_scene.py --mode gui --task scene --cuboids 3 --cylinders 2 --duration 20
```

창 없이 장면 배치·정착을 확인하려면 `--mode direct`로 바꿉니다. CNN 분류·운반 실행은 아래에서 안내합니다.

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

`--block-x 0.48 --block-y -0.025 --block-yaw -0.35`를 추가하면 물체 초기 위치·회전을 바꿀 수 있습니다. 값은 미터·라디안 단위이며, 검증 조건과 결과는 `docs/experiments.csv`에 기록합니다.

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

전체 자동 테스트는 아래 명령으로 실행합니다. 공개 파일만으로 구성한 환경에서는58개가 통과했으며, 개인 모델 파일이 없는 환경에서는 모델을 쓰는 일부 테스트가 건너뛰어집니다.

```bash
.venv/bin/python -m unittest discover -s tests
```

## 집기 실패 후 다시 확인하기

여러 물체 운반 실행에는 기본으로 **최초 시도 + 추가 시도 최대 2회**가 적용됩니다. 한 물체를 옮기는 작업 안에서 후보 보류에 따른 재촬영과 집기 재시도가 이 한도를 함께 사용하며, 운반에 성공하면 다음 작업의 횟수를 새로 셉니다. `--max-retries 0`을 붙이면 첫 실패에서 중단합니다.

- **후보가 모두 보류됨:** 팔을 대기 자세에 두고 0.5초 기다린 뒤 다시 촬영합니다. 계속 보류되면 한도에서 종료합니다.
- **집기 실패:** 이전 카메라 좌표의 접근 높이로 내려 손을 열고, 대기 자세로 복귀한 뒤 새 영상으로 좌표·폭을 다시 구합니다.
- **운반 중 낙하:** 손가락 접촉 상실과 상대 위치 변화로 감지하면 현재 이동을 중단하고 복귀합니다. 뒤쪽에 있으면 바깥쪽 호를 따라 앞쪽으로 돌아옵니다.
- **재시도 대상 확인:** 이전 영상 중심의 3cm 이내에 유효한 후보가 하나 있을 때만 다시 집습니다. 대상이 구역 밖으로 떨어졌거나 후보가 여러 개여서 구별되지 않으면 종료합니다. 재시도 장면은 가까운 후보를 찾기 위해 전체 후보를 분류할 수 있습니다.
- **즉시 중단:** 충돌·관절 제한·목표 자세 미도달·잘못된 상자 도착·복귀 오류는 자동으로 다시 시도하지 않습니다.

잡는 힘과 파지 방식은 기존 설정을 사용합니다. 마지막 집기 실패에서 한도를 소진했을 때도 손을 열고 복귀한 뒤 종료합니다. `sort_result.json`의 `picks`에는 실패한 시도도 남고, `recoveries`에는 원인·복귀 결과·재시도 횟수가 기록됩니다.

실제 첫 집기를 실패시킨 최소·최대 크기 4조건과 랜덤 혼합 1장면에서는 복귀 후 총 9개 물체를 옮겼습니다. 뒤쪽 두 상자 방향의 낙하 실험 2개는 복귀·재촬영 후 대상이 촬영 구역 밖에 있어 종료했습니다. 검증 범위는 이 조건들에 제한됩니다.

실제 손가락을 열어 실패를 만드는 자동 테스트는 아래 명령으로 실행합니다. 이 테스트는 창 없이 실행되며, 테스트에서만 모터 명령을 바꿔 실패를 만듭니다.

```bash
.venv/bin/python -m unittest discover -s tests -p test_grasp_retry.py
```

## ROS2로 실행하기

ROS2 실행은 세 노드가 역할을 나눕니다. **시뮬레이션 노드**가 촬영과 실제 접촉 동작을 맡고, **인식 노드**가 기존 CNN·좌표 계산을 사용하며, **작업 관리 노드**가 대상 선택·운반 순서·재촬영·재시도를 지시합니다. 기존 Python 실행 명령도 그대로 사용할 수 있습니다.

```text
작업 관리: 촬영 요청
    → 시뮬레이션: RGB·깊이·빈 테이블 기준 발행
    → 인식: 영역 분리·필요한 후보만 CNN·좌표 계산
    → 작업 관리: 한 대상 선택·운반 요청
    → 시뮬레이션: 접촉 집기·운반·도착 판정·복귀
    → 작업 관리: 완료 확인 후 다시 촬영
```

### 빌드와 장면 준비

Ubuntu의 **ROS2 Jazzy·Python 3.12·colcon** 환경에서 검증했습니다. 프로젝트 루트에서 실행합니다. ROS2의 시스템 모듈과 기존 PyBullet·torch를 함께 사용하도록 가상환경에 `--system-site-packages`를 적용합니다. 기존 가상환경이 있으면 설치한 패키지를 유지하면서 이 설정을 갱신할 수 있습니다.

```bash
source /opt/ros/jazzy/setup.bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements-sim.txt -r requirements-inference.txt
.venv/bin/python /usr/bin/colcon build --packages-up-to shape_sort_ros --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash
ros2 launch shape_sort_ros shape_sort.launch.py
```

기본값은 직육면체3개·수직 원기둥2개, 랜덤 크기, GUI입니다. 창을 연 뒤 **장면과 CNN이 준비된 상태에서 시작 요청을 기다립니다.** `checkpoints/shape_cnn_v1/`에는 기존 학습 결과가 있어야 합니다. 학습·전처리 설정인 `best_params.json`은 Git에 포함하며, 가중치 `best_model.pt`와 데이터는 제외합니다. 설정 파일만으로 모델을 실행할 수는 없으므로 학습한 가중치는 따로 준비해야 합니다.

별도 터미널에서 시작·상태 확인·정지를 할 수 있습니다. 매 터미널에서 먼저 환경을 불러옵니다.

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 service call /shape_sort/start std_srvs/srv/Trigger '{}'
ros2 topic echo /shape_sort/task_status
```

정지할 때는 다음 명령을 사용합니다.

```bash
ros2 service call /shape_sort/stop std_srvs/srv/Trigger '{}'
```

**시작 서비스의 응답은 접수 여부입니다.** 실제 성공은 `/shape_sort/task_status`의 `stage: FINISHED`, `completed: true`, `success: true`와 결과 파일로 확인합니다. 정지는 다음 물리 스텝에서 현재 이동을 중단하며, 자동으로 물체를 놓거나 팔을 복귀시키지는 않습니다. 중단·완료 후 새 장면은 Ctrl+C로 노드를 끝내고 다시 실행합니다. 작업이 끝나도 상태를 확인할 수 있도록 노드는 유지됩니다.

준비 후 바로 운반을 시작하거나 창 없이 검사하려면 다음처럼 실행합니다.

```bash
ros2 launch shape_sort_ros shape_sort.launch.py auto_start:=true mode:=direct cuboids:=3 cylinders:=2 seed:=1
```

`mode:=gui`는 화면 실행, `size_mode:=fixed`는 기존 기본 크기, `max_retries:=0`은 첫 실패에서 중단입니다. 기본 재시도2회와 실패 구분은 기존 실행과 같습니다. `output_dir:=/원하는/절대경로`로 저장 위치를 지정할 수 있습니다. 기본은 `outputs/ros2-날짜시간/`이며, 관측·crop·CNN 점수·실패 시도·접촉 로그·최종 결과가 저장됩니다.

촬영마다 `scan_id`, 명령마다 `command_id`를 붙여 현재 기다리는 결과만 사용합니다. 깊이 메시지는 미터가 아니라 기존 카메라의 **0~1 depth buffer**이며, 인식 단계에서 기존 함수로 변환합니다. ROS2 연결은 구성요소 사이의 통신이며, 실제 로봇 연결이나 MoveIt 경로 계획을 추가한 것은 아닙니다.

### ROS2 검증 실행

위 빌드와 `source` 후 다음 명령은 기존 회귀 테스트와 ROS 메시지·재시도 정책·실제 세 노드 통합 테스트를 실행합니다. 별도로 실행한 장면과 섞이지 않도록 통합 테스트는 독립 ROS 도메인을 사용합니다.

```bash
SHAPE_SORT_ROS_INTEGRATION=1 .venv/bin/python -m unittest discover -s tests
```

ROS2 통합에서는 기본 크기 혼합2개와 랜덤 크기 혼합5개에서 분류·접촉 운반·최종 정착을 확인했습니다. 잘못된 명령 거절과 이동 중 정지도 확인했습니다. 이는 선택한 장면의 검증이며, 모든 배치의 성공률은 아닙니다. 재시도 정책 테스트와 기존 실제 실패·복구 테스트는 별도로 유지합니다.

## 파일 구조와 역할

기존 실행과 공통 계산은 `scripts/`, ROS2 패키지는 `src/`, 자동 검증은 `tests/`, CNN 학습 코드는 `notebooks/`에 있습니다.

```text
shape-pick-sort-robot/
├── README.md                   프로젝트 소개·설치·실행 방법
├── requirements-sim.txt        시뮬레이션 패키지
├── requirements-inference.txt  CPU CNN 추론 패키지
├── checkpoints/shape_cnn_v1/
│   └── best_params.json        선택한 학습 설정·모델·전처리 정보
├── src/
│   ├── shape_sort_interfaces/  노드끼리 주고받을 메시지·서비스 형식
│   │   ├── msg/               관측·객체 후보·인식 결과·작업 상태
│   │   ├── srv/RobotCommand.srv  촬영·운반·복구 등 명령 접수
│   │   ├── CMakeLists.txt     메시지·서비스 코드 생성
│   │   └── package.xml       인터페이스 의존성
│   └── shape_sort_ros/
│       ├── shape_sort_ros/
│       │   ├── __init__.py
│       │   ├── simulation_node.py    장면·촬영·실제 접촉 동작
│       │   ├── perception_node.py    CNN·영상 좌표 추정
│       │   ├── task_manager_node.py  대상 선택·순서·재시도
│       │   └── protocol.py           영상·후보 변환과 입력 검사
│       ├── launch/shape_sort.launch.py  세 노드 함께 실행
│       ├── resource/shape_sort_ros  ROS 패키지 등록 표시
│       ├── setup.py           노드와 공통 Python 코드 설치
│       ├── setup.cfg          노드 실행 파일 설치 위치
│       └── package.xml        노드 패키지 의존성
├── scripts/
│   ├── multi_object_scene.py   배치·카메라 인식·운반·실패 대응
│   ├── contact_grasp_probe.py  공통 로봇 제어·접촉 검증·단일 물체 실험
│   ├── rgbd_camera.py          깊이 영역 분리·좌표·방향·폭 계산
│   ├── cnn_inference.py        학습 모델 복원·이미지 분류
│   ├── collect_cnn_data.py     학습 이미지 수집·데이터 분할
│   └── terminal_ko.py          터미널 단계·오류의 한국어 표시
├── tests/                     인식·집기·재시도·데이터 등의 자동 검증
├── notebooks/
│   └── shape_cnn_colab.ipynb   CNN 학습·랜덤 탐색·성능 시각화
└── docs/
    ├── DECISIONS_KO.md         선택한 방식·다른 후보·이유
    ├── FOLLOW_UP_KO.md         후속 후보와 보류한 시점
    ├── experiments.csv        검증 조건·결과 기록
    └── images/cnn/            대표 CNN 결과 그래프
```

`multi_object_scene.py`가 전체 순서를 진행하고, `rgbd_camera.py`의 좌표와 `contact_grasp_probe.py`의 제어·검증을 사용합니다.



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
- [x] 전체 작업과 제한된 실패 대응 통합
- [ ] ROS2 연동 및 시연·발표 정리

## 평가 계획

**모델:** 학습·검증 loss, 분류 정확도, 클래스별 혼동행렬, 조건별 오분류, 설정별 실험 비교.

**시스템:** 첫 집기 성공률, 재시도 포함 성공률, 운반 낙하, 올바른 상자 도착률. 필요하면 위치 추정 오차와 처리 시간도 측정합니다.

성능 수치와 시연 영상은 실제 실험 후 추가합니다.

## 기술 구성

Python · PyBullet · PyTorch · NumPy · OpenCV · ROS2

시뮬레이션·추론은 Ubuntu, 모델 학습은 Google Colab을 사용할 계획입니다. 첫 실행은 Ubuntu 24.04.5 / Python 3.12.3 / PyBullet 3.2.7에서 검증했습니다. ROS2 Jazzy의 명령과 Python import를 확인했으며, 프로젝트 통신 연결은 후속 단계입니다.

## 문서

- [의사결정 기록](docs/DECISIONS_KO.md)
- [실험 기록 양식](docs/experiments.csv)

작은 단계별로 구현하고 검증 결과를 실험 기록에 남깁니다.

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

원본 관측·crop·클래스 점수·접촉 로그는 `outputs/cnn-local-validation/`에 로컬로 보관합니다. 현재는 원인별 제한된 재시도와 ROS2 통신 실행을 연결했습니다.

사진·zip·가중치·전체 실행 결과와 개인 학습 정리는 Git에서 제외합니다. [Colab 학습 코드](notebooks/shape_cnn_colab.ipynb)와 README에 사용하는 대표 그래프4장만 저장소에 남깁니다. 원본 실험 기록은 개인 Drive와 로컬 `outputs/colab-result-review/`에 보관합니다.

데이터 수집을 다시 실행하려면 아래 명령을 사용합니다. 실행마다 새 출력 폴더를 만들고, 사진·라벨·분할을 저장한 zip을 개인 Drive로 옮겨 Colab에서 학습합니다. 노트북 기본값은12회·최대20epoch·조기종료4epoch이며, **이번 본 학습은30회·30epoch·6epoch로 늘린 결과**입니다.

```bash
.venv/bin/python -m pip install -r requirements-sim.txt
.venv/bin/python scripts/collect_cnn_data.py --scenes 2000 --seed 2026
```
