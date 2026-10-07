# 한 번 촬영한 RGB·깊이·빈 테이블 기준을 같은 관측 번호로 묶는다.
std_msgs/Header header
uint64 scan_id
sensor_msgs/Image rgb
# depth와 empty_depth는 64FC1 OpenGL buffer(0~1)이며 미터 거리가 아니다.
sensor_msgs/Image depth
sensor_msgs/Image empty_depth
float64[16] view
float64[16] projection
# 기존 카메라 설정 사전. 생성 물체의 종류·위치 정답은 포함하지 않는다.
string camera_json
bool all_candidates
bool random_size
float64[4] size_range_m
