# 문서 읽는 순서
이 문서는 가벼운 SDD와 실험 기록을 위한 안내다. 제품 코드는 아직 없다.

1. 저장소 루트 README.md: 프로젝트 소개와 개발 상태.
2. superpowers/specs/2026-10-04-shape-sort-design.md: 요구사항 ID·정보 경계·단계 통과 기준.
3. experiments.csv: 실험 기록 헤더. 현재 결과 행이 없으며 성능을 뜻하지 않는다.

로컬 전용 HANDOFF_FULL_KO.md는 전체 배경 인계문, PROGRESS.md는 환경·진행 기록이다. 개인 정보가 포함된 두 파일은 Git 추적 대상에서 제외하며 공개 저장소에는 포함하지 않는다.

## 실험 기록 규칙
- experiment_kind: grasp_oracle / perception / model / system / negative_control 등 실험 역할을 표시.
- pose_source는 oracle/camera를 명시. 두 모드 결과를 같은 성공률로 합치지 않는다.
- true_class와 predicted_class, lift/drop, 예측상자/정답상자 도착을 분리한다.
- attempt_index는 최초=0, 재시도=1/2. 작업당 최대3시도.
- 평가상 적용되지 않거나 측정되지 않은 값은 빈칸으로 두고 notes에 이유를 쓴다. 빈칸을 실패0으로 해석하지 않는다.
- 성공/실패 값을 기록할 때 사용한 config_ref와 로그/영상 artifact_ref를 남긴다.
- config_ref는 물체·카메라·힘·마찰·시간·판정 설정을 재현 가능한 파일로 가리킨다.
- 데이터의 실제 이미지/geometry/학습 전체 설정은 추후 산출물에 저장하며 CSV에 모두 복제하지 않는다.
- recorded_at은 KST offset을 포함한 시각으로 기록한다. sim_duration과 wall_duration을 구분한다.
- 현재 CSV는 헤더뿐이며 실행 결과를 만들지 않았다.

## 문서 운영
배경을 매번 재작성하지 않는다. 범위/결정은 spec, 실험은 CSV/산출물, 현재 상태는 progress에 기록한다.
Ubuntu 환경 확인 후 첫 구현 계획은 docs/superpowers/plans/ 아래에 작성한다.
큰 미래 단계의 함수/설정까지 미리 고정하지 않는다. 다음 검증 단계의 인터페이스·명령·테스트부터 결정한다.
기존 Windows 인증/세션 DB를 이전 대상으로 삼지 않는다.
