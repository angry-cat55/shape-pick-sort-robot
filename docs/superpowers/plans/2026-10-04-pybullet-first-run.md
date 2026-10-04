# PyBullet 첫 실행 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 프로젝트 전용 환경에서 PyBullet의 물리 계산과 GUI를 실행하고 사용자가 같은 명령으로 재현한다.

**Architecture:** 첫 실행은 ROS2와 분리된 단일 Python 프로그램이다. DIRECT와 GUI 모드가 같은 장면을 사용하며, 한 실행 루프만 물리 시간을 진행한다. 이후 접촉 파지와 ROS2 연동은 별도 계획으로 확장한다.

**Tech Stack:** Ubuntu 24.04.5, Python 3.12.3, PyBullet. ROS2 Jazzy는 별도 실행 확인.

**Spec:** ../specs/2026-10-04-shape-sort-design.md

## Global Constraints

- 기존 환경·파일 보존. 프로젝트 `.venv`에만 Python 의존성 설치.
- 로컬 물리·추론은 CPU부터 시작. GPU 드라이버 변경 없음.
- 강제 부착 및 실행 중 관절·물체 순간이동으로 성공 만들기 금지.
- 첫 장면 실행은 집기·카메라 인식·CNN·ROS2 통합 성공을 의미하지 않는다.
- 사용자 최신 결정: ROS2 연동은 필수 범위. 학습 데이터는 로컬 카메라 경로 확인 후 생성.

## Review Focus

- DISPLAY가 없는 환경: DIRECT 모드를 실행할 수 있어야 한다.
- GUI 종료·Ctrl+C: 연결을 해제하고 종료한다.
- 설치 실패: 기존 시스템 Python을 수정하지 않고 실패와 원인을 기록한다.
- 시뮬레이션 시간: 1/240초 고정 step으로 진행하며 wall time과 구분한다.
- 물체가 바닥을 통과하거나 멈춰 있음: 초기·최종 높이를 비교해 물리 계산을 확인한다.

## Task 1: 전용 환경과 재현 가능한 첫 장면

**Files:** Create `requirements-sim.txt`, `scripts/pybullet_first_run.py`, `docs/PYBULLET_START_KO.md`; Modify `README.md`, local `docs/PROGRESS.md`.

**Interfaces:** CLI `python scripts/pybullet_first_run.py --mode direct|gui --steps 480`. DIRECT 기본값. 출력은 mode, step 수, sim 시간, 물체 초기·최종 z, 높이 검사 결과. GUI는 제한된 step 실행 후 종료하며 사용자는 step 수를 늘릴 수 있다.

- [x] `python3 -m venv .venv`로 생성하고 `.venv/bin/python -m pip --version` 확인.
- [x] `.venv`에 PyBullet 설치, 실제 설치 버전을 `requirements-sim.txt`에 고정.
- [x] 단일 프로그램에 plane, 고정 기반 Panda, 40×30×40 mm/0.05 kg 직육면체 생성. 직육면체는 Panda에서 떨어진 바닥 위 x=0.5, y=0, z=0.2에 배치. 중력 z=-9.81, step=1/240초. 초기 배치 후 pose reset/constraint 없이 실행.
- [x] DIRECT 480 step으로 물체가 내려와 바닥에 정착하는지 확인: 최종 중심 z는 0.02±0.005 m. Panda joint/link 이름도 조회해 출력하되 TCP를 검증했다고 주장하지 않는다.
- [x] GUI를 별도 실행해 창 생성 여부 확인. 창이 사용자 화면에 정상 표시되는지는 사용자 관찰로 확인하며 도구 실행 성공만으로 시각 확인을 대신하지 않는다.
- [x] 연결 해제 및 KeyboardInterrupt 종료 경로 확인. GUI 부재는 DIRECT 결과와 별도로 기록.
- [x] 설치·DIRECT·GUI·ROS source 명령과 각 의미를 시작 안내에 작성. 실제 실행 결과를 진행 기록에 반영.

## 실행 기록

- PyBullet 3.2.7 설치. CPython 3.12용 binary-only 설치는 후보 없음으로 실패했고, 기존 컴파일러/헤더로 소스 빌드 성공.
- DIRECT: 480 step, sim 2초, 중심 z=0.0199888094m, 바닥 접촉점 4개, PASS.
- GUI: 7200 step, sim 30초, wall 약 30.96초, PASS. 사용자가 Panda/빨간 물체 표시 확인.
- unittest 4개 RED→GREEN, pip check 통과, SIGINT 종료 코드 130 확인.
- Ruling: 사용자 승인 범위의 파일을 현재 프로젝트 경로에서 바로 실행할 수 있도록 별도 worktree 대신 작업 브랜치를 사용. 기존 문서 변경은 보존하고 자동 commit/push는 하지 않음.
- 작성 후 자체 검토 수행. 초기 자세 설정과 이후 모터 유지, 물체 순간이동/부착 없음, 실제 접촉·속도 측정과 GUI 관찰 여부를 구분해 확인.

## 이후 단계

첫 실행 다음은 테이블·TCP 확인 및 고정 직육면체 접촉 파지이다. 첫 파지 검증 전 대량 데이터 수집은 하지 않는다. ROS2 통신 실험과 최종 영상→인식→작업 상태 연결은 필수로 유지한다. 이번 계획은 R11/R12와 환경·작은 장면 단계만 다루며 R01–R10 전체 구현 계획이 아니다.
