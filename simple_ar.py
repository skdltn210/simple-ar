import os
import cv2 as cv
import numpy as np
import colorsys
from scipy.spatial.transform import Rotation

if __name__ == "__main__":
    input_file = "checkerboard.avi"  # 비디오 파일의 경로 지정
    video_scale = 1
    board_pattern = (10, 7)
    board_cellsize = 0.021
    board_criteria = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

    K = np.array([[889.12722828, 0, 958.85335127],
                  [0, 887.90752657, 538.57413936],
                  [0, 0, 1]])
    dist_coeff = np.array([-0.01573471, 0.00598868, 0.00133999, -0.00048363, -0.00065088])

    # Open a video
    if os.path.exists(input_file):
        video = cv.VideoCapture(input_file)
    else:
        raise FileNotFoundError("Video file does not exist: " + input_file)

    assert video.isOpened(), 'Cannot read the given input, ' + input_file

    height, width = video.get(cv.CAP_PROP_FRAME_HEIGHT), video.get(cv.CAP_PROP_FRAME_WIDTH)
    video.set(cv.CAP_PROP_FRAME_HEIGHT, height*video_scale)
    video.set(cv.CAP_PROP_FRAME_WIDTH, width*video_scale)

    obj_points = board_cellsize * np.array([[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])])

    # Open object file
    obj = np.loadtxt("pikachu.xyz")
    obj = obj[obj[:,1].argsort()] # For coloring
    obj_rotation = [0,180,180] # Set default rotation
    obj_rvec = Rotation.from_euler('zyx', obj_rotation[::-1], degrees=True).as_matrix()
    obj_tvec = np.array([0.1, 0.1, 0]) # Set default position
    obj_result = obj @ obj_rvec + obj_tvec

    point_color = [np.array(colorsys.hsv_to_rgb(i/len(obj), 1.0, 1.0)) * 255 for i in range(len(obj))]

    # Run pose estimation
    while True:
        # Read an image from the video
        valid, img = video.read()
        assert valid, "Video is not valid"

        # Estimate the camera pose
        complete, img_points = cv.findChessboardCorners(img, board_pattern, flags=board_criteria)
        if complete:
            ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)

            # Draw the object on the image
            point, _ = cv.projectPoints(obj_result, rvec, tvec, K, dist_coeff)
            for p, p_c in zip(point, point_color):
                cv.circle(img, np.int32(p.flatten()), 10, p_c, -1) # Trick to draw surface

            # Print the camera position
            R, _ = cv.Rodrigues(rvec) # Alternative) scipy.spatial.transform.Rotation
            p = (-R.T @ tvec).flatten()
            info = f'XYZ: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]'
            cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

        # Show the image and process the key event
        cv.imshow('Simple AR', img)

        key = cv.waitKey(1)

        if key == ord(' '):
            key = cv.waitKey()
        if key == 27: # ESC
            break

        # Control object position
        if key != -1 and key in b"wasdqe":
            if key == ord('w'):
                obj_tvec[1] -= 0.01
            if key == ord('s'):
                obj_tvec[1] += 0.01
            if key == ord('a'):
                obj_tvec[0] -= 0.01
            if key == ord('d'):
                obj_tvec[0] += 0.01

            if key == ord('q'):
                obj_rotation[2] -= 5
            if key == ord('e'):
                obj_rotation[2] += 5

            obj_rvec = Rotation.from_euler('zyx', obj_rotation[::-1], degrees=True).as_matrix()
            obj_result = obj @ obj_rvec + obj_tvec

    video.release()
    cv.destroyAllWindows()
