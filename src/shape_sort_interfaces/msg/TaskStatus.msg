std_msgs/Header header
# source는 simulation 또는 task_manager, stage는 기계 판정용 식별자다.
string source
uint64 command_id
uint64 scan_id
string stage
bool completed
bool success
string failure_reason
# 평가·시도 기록. task_manager는 정답 좌표를 잡기 계획에 사용하지 않는다.
string result_json
