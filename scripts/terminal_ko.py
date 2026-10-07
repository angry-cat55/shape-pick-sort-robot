"""터미널 표시만 한글로 바꾼다. 저장 로그의 영문 식별자는 그대로 유지한다."""

import argparse
import sys

STAGES = {
    'RECOVERY_LOWER': '집기 실패 후 원래 접근 높이로 내려놓기',
    'RECOVERY_RELEASE': '복구를 위해 손가락 열고 기다리기',
    'RECOVERY_CLEARANCE': '복귀를 위한 안전 높이로 후퇴',
    'RECOVERY_ARC': '로봇 바깥쪽 호를 따라 앞쪽으로 복귀',
    'REOBSERVE_WAIT': '보류된 후보를 다시 촬영하기 전 대기',
    'SETUP': '장면 준비',
    'SETTLE': '물체 안정화 대기',
    'OBSERVE': '카메라 촬영·좌표 계산',
    'APPROACH': '물체 위로 접근',
    'DESCEND': '집을 높이로 내려가기',
    'CLOSE': '손가락 닫기',
    'LIFT': '물체 들어 올리기',
    'VERIFY_LIFT': '물체 들림·유지 확인',
    'CLEARANCE': '안전한 운반 높이로 이동',
    'CARRY_ARC': '로봇 바깥쪽 호를 따라 운반',
    'TRANSPORT_HALF': '낙하 실험의 중간 지점으로 운반',
    'TRANSPORT': '상자로 운반',
    'BIN_ABOVE': '상자 위로 이동',
    'BIN_PLACE': '상자 바닥 가까이 내려놓기',
    'BIN_RELEASE': '상자에서 손가락 열기',
    'BIN_RETREAT': '상자 위로 후퇴',
    'ARRIVAL_SETTLE': '상자에 놓인 물체 안정화 대기',
    'VERIFY_ARRIVAL': '물체의 상자 도착 확인',
    'VERIFY_FINAL_ARRIVAL': '전체 물체의 상자 정착 최종 확인',
    'PLACE': '원래 위치에 내려놓기',
    'RELEASE': '손가락 열어 물체 놓기',
    'RETREAT': '물체 위로 후퇴',
    'VERIFY_PLACE': '내려놓은 물체 확인',
    'RETURN_WAIT': '촬영용 대기 자세로 복귀',
    'WAIT_SETTLE': '대기 자세에서 안정화 대기',
    'FORCED_OPEN': '대조 실험: 손가락 열어 낙하 유도',
    'FORCED_TRANSPORT_DROP': '대조 실험: 운반 중 낙하 확인',
}
NAMES = {
    'direct': '화면 없이 실행',
    'gui': '창으로 실행',
    'oracle': '시뮬레이터 정답 좌표',
    'camera': '카메라 추정 좌표',
    'cuboid': '직육면체',
    'cylinder': '원기둥',
    'normal': '정상 집기',
    'miss': '빗나간 집기',
    'open': '집은 뒤 손가락 열기',
    'support': '받침대 지지 대조',
    'transport': '상자 운반',
    'transport_drop': '운반 중 낙하 대조',
    'wrong_bin': '다른 상자 도착 대조',
    'camera_empty': '빈 장면 탐지 대조',
}
FAILURES = {
    'task_stopped': '사용자 요청으로 동작을 중단함',
    'communication_timeout': 'ROS2 응답 제한 시간이 지나 작업을 중단함',
    'simulation_unavailable': '시뮬레이션 명령 서비스에 연결할 수 없음',
    'initialization_failed': 'ROS2 시뮬레이션 초기화에 실패함',
    'initial_scene_not_settled': '초기 배치가 안정적으로 정착하지 못함',
    'perception_failed': '카메라 데이터 처리 또는 CNN 인식에 실패함',
    'command_rejected': '시뮬레이션이 명령을 거절함',
    'command_service_failed': '시뮬레이션 명령 전달에 실패함',
    'retry_limit_reached': '허용한 재시도 횟수를 모두 사용함',
    'retry_target_not_found': '이전 영상 중심 근처에서 재시도 대상을 찾지 못함',
    'retry_target_ambiguous': '이전 영상 중심 근처에 후보가 여러 개여서 재시도 대상을 구별하지 못함',
    'recovery_failed': '안전 복귀 중 오류가 발생하여 중단함',
    'task_iteration_limit': '작업의 최대 반복 횟수에 도달함',
    '': '없음',
    'arm_table_collision': '팔이 작업대와 충돌함',
    'arm_bin_collision': '팔이 상자와 충돌함',
    'ik_joint_limit': '역기구학 결과가 관절 허용 각도를 벗어남',
    'tcp_not_reached': '손끝이 목표 자세에 도착하지 못함',
    'wait_pose_not_reached': '촬영용 대기 자세에 도착하지 못함',
    'evaluation_target_unmatched': '추정 대상과 검증 대상을 대응하지 못함',
    'lift_not_verified': '물체가 들린 상태를 확인하지 못함',
    'drop_during_transport': '운반 중 물체가 떨어짐',
    'drop_during_motion': '이동 중 물체가 떨어짐',
    'drop_after_open': '손가락을 연 뒤 물체가 떨어짐',
    'wrong_bin_arrival': '지정한 상자가 아닌 다른 상자에 도착함',
    'arrival_not_verified': '상자 도착·정착을 확인하지 못함',
    'classification_wrong_bin': '분류 오류로 실제 종류와 다른 상자에 놓임',
    'final_arrival_not_verified': '작업 종료 후 전체 물체의 정착을 확인하지 못함',
    'no_valid_target': '집을 수 있는 유효한 후보가 없음',
    'no_object': '물체를 탐지하지 못함',
    'multiple_objects': '단일 물체 검사에서 여러 영역이 탐지됨',
    'invalid_depth': '깊이 영상에 잘못된 값이 있음',
    'insufficient_top_surface': '좌표를 계산할 윗면 점이 부족함',
    'unsupported_geometry': '물체 폭·높이가 지원 범위를 벗어남',
    'object_at_roi_edge': '물체가 촬영 구역 경계에서 잘렸을 수 있음',
    'incomplete_or_merged_surface': '윗면이 잘렸거나 여러 물체 영역이 붙었을 수 있음',
    'low_class_confidence': 'CNN 분류 점수가 기준보다 낮음',
    'not_evaluated': '다른 후보를 선택하여 이번에는 검사하지 않음',
    'supported_by_pedestal': '물체가 손가락 대신 받침대에 지지됨',
    'place_not_verified': '원래 위치에 내려놓기를 확인하지 못함',
    'placement_exhausted': '간격을 확보한 물체 배치를 만들지 못함',
    'ambiguous_annotation': '영역의 정답 종류를 명확히 구분하지 못함',
    'cropped_at_roi_edge': '물체가 촬영 구역 경계에서 잘림',
    'hidden_or_merged_object': '물체가 가려졌거나 영역이 붙음',
    'possible_top_occlusion': '더 높은 물체가 윗면을 가릴 수 있어 보류함',
    'non_circular_top': '윗면이 원형 조건을 만족하지 않음',
    'unsupported_shape': '지원하지 않는 도형 종류',
    'tilted_object': '물체가 기울어져 있음',
    'object_outside_region': '물체가 생성 구역 밖으로 나감',
    'insufficient_gap': '물체 사이 간격이 부족함',
    'settle_timeout': '정해진 시간 안에 물체가 안정화되지 않음',
    'need_both_classes_for_split': '각 데이터 분할에 두 도형 종류가 필요함',
}


def stage_name(stage):
    if stage in STAGES:
        return STAGES[stage]
    # 같은 이동 단계에 붙는 도착 확인·안정화 접미사를 함께 처리한다.
    for suffix, label in [
        ('_SETTLE', ' 후 안정화 대기'),
        ('_FINAL', ' 목표 도착 확인'),
    ]:
        if stage.endswith(suffix):
            return stage_name(stage[: -len(suffix)]) + label
    return '동작 단계 확인 필요'


def failure_name(reason):
    code, separator, detail = reason.partition(':')
    message = FAILURES.get(code, code)
    if separator:
        message += ': ' + detail.replace('position=', '위치 오차(m)=').replace(
            'orientation=', '방향 오차(rad)='
        )
    return message


def verdict(value):
    return '성공' if value else '실패'


OPTION_HELP = {
    '--mode': '실행 방식: direct는 화면 없이, gui는 창으로 실행',
    '--steps': '물리 계산 횟수',
    '--scenario': '정상 동작 또는 실패 대조 실험 선택',
    '--pose-source': '좌표 출처: oracle은 정답, camera는 카메라 추정',
    '--object-shape': '도형 종류: cuboid는 직육면체, cylinder는 원기둥',
    '--block-x': '물체 초기 X좌표(m)',
    '--block-y': '물체 초기 Y좌표(m)',
    '--camera-angle': '카메라의 위쪽 기울기 각도(도)',
    '--output-dir': '새 결과 저장 폴더',
    '--cuboids': '직육면체 개수',
    '--cylinders': '원기둥 개수',
    '--seed': '랜덤 배치를 재현할 번호',
    '--model-dir': '학습한 모델과 설정 파일이 있는 폴더',
    '--scenes': '수집할 독립 장면 수',
    '--min-width-cm': '물체 폭의 최솟값(cm)',
    '--max-width-cm': '물체 폭의 최댓값(cm)',
    '--min-height-cm': '물체 높이의 최솟값(cm)',
    '--max-height-cm': '물체 높이의 최댓값(cm)',
}


class KoreanArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._optionals.title = '실행 옵션'
        self._positionals.title = '위치 인자'
        self._actions[0].help = '실행 방법을 표시하고 종료합니다'

    def add_argument(self, *args, **kwargs):
        # 명령에 쓰는 옵션 이름은 유지하고, 설명이 없던 옵션에도 한글 도움말을 붙인다.
        if args and args[0] in OPTION_HELP:
            kwargs.setdefault('help', OPTION_HELP[args[0]])
        return super().add_argument(*args, **kwargs)

    def format_usage(self):
        return super().format_usage().replace('usage:', '사용법:')

    def format_help(self):
        return super().format_help().replace('usage:', '사용법:')

    def error(self, message):
        # argparse의 기본 입력 오류도 한글로 표시하되 옵션 이름·입력값은 보존한다.
        for original, translated in [
            ('unrecognized arguments:', '알 수 없는 옵션:'),
            ('invalid choice:', '허용하지 않는 선택값:'),
            ('(choose from ', '(허용값: '),
            ('argument ', '옵션 '),
            ('invalid int value:', '정수로 입력해야 합니다:'),
            ('invalid float value:', '숫자로 입력해야 합니다:'),
            ('expected one argument', '값을 하나 입력해야 합니다'),
            ('the following arguments are required:', '필수 옵션을 입력하세요:'),
        ]:
            message = message.replace(original, translated)
        self.print_usage(sys.stderr)
        self.exit(2, f'{self.prog}: 입력 오류: {message}\n')
