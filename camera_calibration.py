import numpy as np
import cv2

# 체스보드 크기와 셀 크기
chessboard_size = (10, 7)
cell_size = 0.021  # 미터 단위

# 체스보드의 3D 좌표 생성
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * cell_size

# 객체 포인트 및 이미지 포인트 저장
objpoints = []  # 실제 세계 좌표
imgpoints = []  # 이미지 상의 2D 포인트

# 동영상 파일 경로
video_path = 'checkerboard.avi'

# 동영상 파일 불러오기
cap = cv2.VideoCapture(video_path)

# 코너 찾기 및 코너 서브픽셀 검색을 위한 기준 정의
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

count = 0  # 캘리브레이션에 사용된 이미지 수 초기화

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 체스보드 코너 찾기
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # 코너가 발견된 경우
    if ret == True:
        objpoints.append(objp)

        # 정확한 코너 위치 찾기
        corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners_subpix)

        # 코너 그리기
        cv2.drawChessboardCorners(frame, chessboard_size, corners_subpix, ret)
        count += 1  # 캘리브레이션에 사용된 이미지 수 증가

        # 이미지에 캘리브레이션에 사용된 이미지 수 표시
        cv2.putText(frame, f'Used Images: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Chessboard', frame)
    key = cv2.waitKey(1)
    if key == 27:  # ESC 키를 누르면 종료
        break

# 카메라 캘리브레이션
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 카메라 내부 매개 변수 출력 (행렬 형태)
print("카메라 내부 매개 변수 (행렬 형태):")
print(mtx)

# 카메라 내부 매개 변수 출력 (fx, fy, cx, cy, ...)
fx = mtx[0, 0]
fy = mtx[1, 1]
cx = mtx[0, 2]
cy = mtx[1, 2]

# RMSE 값 출력
rmse = ret
print("\n카메라 내부 매개 변수:")
print(f"(fx, fy, cx, cy, ..., rmse): ({fx}, {fy}, {cx}, {cy}, ..., {rmse})")

# 렌즈 왜곡 매개 변수 출력
print("\n렌즈 왜곡 매개 변수:")
print(dist)

# 카메라 캘리브레이션 결과 저장
np.savez('calibration_result.npz', mtx=mtx, dist=dist)

# 종료
cap.release()
cv2.destroyAllWindows()
