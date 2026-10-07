# 물체 한 후보의 영상 추정값. center는 월드 XY, 길이는 미터, yaw는 라디안.
bool valid
string failure_reason
string object_shape
float64[2] center_xy_m
float64 top_z_m
float64 width_m
float64 yaw_rad
uint32 top_pixels
bool cnn_evaluated
float64 class_score
# 부가 영상 측정·분류 점수 기록. mask와 시뮬레이터 정답은 보내지 않는다.
string details_json
