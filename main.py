import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template_string, Response
import tempfile
import os
import json
import threading
import time
import math

app = Flask(__name__)

# Configuration
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
CALIBRATION_FRAMES_NEEDED = 15
FRAME_SKIP = 3

# OV2311 specific resolutions
RESOLUTIONS = [
    (640, 480),
    (800, 600),
    (1280, 960),
    (1600, 1300),
]

# Global variables
capture_running = False
captured_frames = []
detected_frames = []
detection_positions = []  # Store where boards were detected
capture_resolution = (800, 600)
camera_cap = None
last_detection_image = None
detection_count = 0
frame_count = 0
last_detected = False
camera_lock = threading.Lock()
brightness_adjustment = 0
contrast_adjustment = 1.0


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def adjust_brightness_contrast(frame, brightness=0, contrast=1.0):
    """Adjust brightness and contrast of frame"""
    adjusted = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
    return adjusted


def get_opencv_version():
    """Get OpenCV version as tuple"""
    version = cv2.__version__.split('.')
    return tuple(int(x) for x in version[:3])


def create_charuco_board_and_detector():
    """Create ChArUco board and detector for OpenCV 4.10+"""
    squares_x = 7
    squares_y = 5
    square_length = 0.04  # 4cm
    marker_length = 0.02  # 2cm

    version = get_opencv_version()
    print(f"OpenCV version: {'.'.join(map(str, version))}")

    # Get dictionary
    try:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        print("✓ Got ArUco dictionary")
    except:
        dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        print("✓ Got ArUco dictionary (old API)")

    # Create board
    try:
        board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, dictionary)
        print("✓ Created CharucoBoard (new API)")
    except:
        board = cv2.aruco.CharucoBoard_create(squares_x, squares_y, square_length, marker_length, dictionary)
        print("✓ Created CharucoBoard (old API)")

    # Create detector
    detector = None
    charuco_detector = None

    try:
        detector_params = cv2.aruco.DetectorParameters()
        aruco_detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
        charuco_params = cv2.aruco.CharucoParameters()
        charuco_detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)
        print("✓ Created CharucoDetector (OpenCV 4.10+)")
        return board, dictionary, aruco_detector, charuco_detector, "charuco_detector"
    except Exception as e:
        print(f"CharucoDetector not available: {e}")

    try:
        detector_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
        print("✓ Created ArucoDetector (OpenCV 4.7+)")
        return board, dictionary, detector, None, "aruco_detector"
    except Exception as e:
        print(f"ArucoDetector not available: {e}")

    print("✓ Using old API (OpenCV 4.0-4.6)")
    return board, dictionary, None, None, "old"


def detect_and_interpolate_charuco(gray, board, dictionary, detector, charuco_detector, api_type):
    """Detect markers and interpolate CharUco corners"""

    if api_type == "charuco_detector":
        try:
            charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)

            if charuco_ids is not None and len(charuco_ids) > 5:
                return True, marker_corners, marker_ids, charuco_corners, charuco_ids
            return False, None, None, None, None
        except Exception as e:
            print(f"CharucoDetector.detectBoard failed: {e}")
            return False, None, None, None, None

    elif api_type == "aruco_detector":
        try:
            marker_corners, marker_ids, rejected = detector.detectMarkers(gray)

            if marker_ids is None or len(marker_ids) == 0:
                return False, None, None, None, None

            try:
                if hasattr(board, 'matchImagePoints'):
                    charuco_corners, charuco_ids = board.matchImagePoints(marker_corners, marker_ids)
                    if charuco_ids is not None and len(charuco_ids) > 5:
                        return True, marker_corners, marker_ids, charuco_corners, charuco_ids
                else:
                    try:
                        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                            marker_corners, marker_ids, gray, board
                        )
                        if retval and charuco_ids is not None and len(charuco_ids) > 5:
                            return True, marker_corners, marker_ids, charuco_corners, charuco_ids
                    except AttributeError:
                        pass
            except Exception as e:
                print(f"Interpolation failed: {e}")

            return False, None, None, None, None
        except Exception as e:
            print(f"Marker detection failed: {e}")
            return False, None, None, None, None

    else:
        try:
            marker_corners, marker_ids, rejected = cv2.aruco.detectMarkers(gray, dictionary)

            if marker_ids is None or len(marker_ids) == 0:
                return False, None, None, None, None

            retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                marker_corners, marker_ids, gray, board
            )

            if retval and charuco_ids is not None and len(charuco_ids) > 5:
                return True, marker_corners, marker_ids, charuco_corners, charuco_ids

            return False, None, None, None, None
        except Exception as e:
            print(f"Old API detection failed: {e}")
            return False, None, None, None, None


def get_board_center(charuco_corners):
    """Get center point of detected board"""
    if charuco_corners is None or len(charuco_corners) == 0:
        return None
    corners_array = np.array(charuco_corners).reshape(-1, 2)
    center = np.mean(corners_array, axis=0)
    return tuple(center.astype(int))


def detect_charuco_board(frame, board, dictionary, detector, charuco_detector, api_type):
    """Detect ChArUco board in frame"""
    global last_detected, brightness_adjustment, contrast_adjustment, detection_positions

    if frame is None or frame.size == 0:
        return False, frame, None, None

    frame_adjusted = adjust_brightness_contrast(frame, brightness_adjustment, contrast_adjustment)

    if len(frame_adjusted.shape) == 3:
        gray = cv2.cvtColor(frame_adjusted, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame_adjusted
        frame_adjusted = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    detected, marker_corners, marker_ids, charuco_corners, charuco_ids = detect_and_interpolate_charuco(
        gray, board, dictionary, detector, charuco_detector, api_type
    )

    if detected:
        frame_with_detection = frame_adjusted.copy()

        # Draw markers
        if marker_corners is not None and marker_ids is not None:
            cv2.aruco.drawDetectedMarkers(frame_with_detection, marker_corners, marker_ids)

        # Draw CharUco corners
        if charuco_corners is not None and charuco_ids is not None:
            cv2.aruco.drawDetectedCornersCharuco(frame_with_detection, charuco_corners, charuco_ids)

        # Get board center for position tracking
        center = get_board_center(charuco_corners)
        if center is not None:
            # Draw a circle at board center
            cv2.circle(frame_with_detection, center, 10, (255, 0, 255), -1)
            cv2.circle(frame_with_detection, center, 15, (255, 0, 255), 2)

        cv2.putText(frame_with_detection, "BOARD DETECTED!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame_with_detection,
                    f"Markers: {len(marker_ids) if marker_ids is not None else 0} | Corners: {len(charuco_ids)}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        last_detected = True
        return True, frame_with_detection, charuco_corners, charuco_ids, center

    frame_with_text = frame_adjusted.copy()
    cv2.putText(frame_with_text, "NO BOARD DETECTED", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame_with_text, "Adjust brightness/contrast or move board", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    last_detected = False
    return False, frame_with_text, None, None, None


def draw_detection_coverage(frame, positions, frame_shape):
    """Draw coverage map showing where board was detected"""
    if not positions:
        return frame

    overlay = frame.copy()

    # Draw grid
    h, w = frame_shape[:2]
    grid_size = 50

    # Draw previous detection positions
    for pos in positions:
        if pos is not None:
            cv2.circle(overlay, pos, 20, (0, 255, 255), -1)  # Yellow dots
            cv2.circle(overlay, pos, 25, (0, 200, 200), 2)

    # Blend overlay
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    # Draw coverage info
    cv2.putText(frame, f"Coverage: {len(positions)} positions", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return frame


def calculate_fov(camera_matrix, image_size):
    """Calculate FOV from camera matrix"""
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    width, height = image_size

    fov_x = 2 * math.atan(width / (2 * fx)) * (180 / math.pi)
    fov_y = 2 * math.atan(height / (2 * fy)) * (180 / math.pi)
    fov_diag = 2 * math.atan(
        math.sqrt(width ** 2 + height ** 2) / (2 * math.sqrt(fx ** 2 + fy ** 2))
    ) * (180 / math.pi)

    return {
        'horizontal': float(fov_x),
        'vertical': float(fov_y),
        'diagonal': float(fov_diag),
        'aspect_ratio': width / height
    }


def calibrate_camera_new_api(all_charuco_corners, all_charuco_ids, board, image_size):
    """Calibrate camera using OpenCV 4.10+ API or standard cv2.calibrateCamera"""

    # Prepare object points and image points for standard calibration
    obj_points = []  # 3D points in real world space
    img_points = []  # 2D points in image plane

    # Get board's 3D corner positions
    try:
        if hasattr(board, 'getChessboardCorners'):
            board_obj_points = board.getChessboardCorners()
        elif hasattr(board, 'chessboardCorners'):
            board_obj_points = board.chessboardCorners
        else:
            board_obj_points = board.objPoints
    except:
        board_obj_points = board.objPoints if hasattr(board, 'objPoints') else None

    if board_obj_points is None:
        raise ValueError("Cannot get board object points")

    # Build object and image point arrays
    for charuco_corners, charuco_ids in zip(all_charuco_corners, all_charuco_ids):
        obj_pts_frame = []
        img_pts_frame = []

        for i, corner_id in enumerate(charuco_ids):
            obj_pts_frame.append(board_obj_points[corner_id[0]])
            img_pts_frame.append(charuco_corners[i][0])

        obj_points.append(np.array(obj_pts_frame, dtype=np.float32))
        img_points.append(np.array(img_pts_frame, dtype=np.float32))

    # Use standard OpenCV calibration
    print("Running cv2.calibrateCamera()...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )

    return ret, camera_matrix, dist_coeffs, rvecs, tvecs


def calibrate_from_frames(frames, board, dictionary, detector, charuco_detector, api_type):
    """Calibrate camera from frames"""
    all_charuco_corners = []
    all_charuco_ids = []
    frames_used = 0
    image_size = None

    print(f"Processing {len(frames)} frames for calibration...")

    for idx, frame in enumerate(frames):
        if idx % FRAME_SKIP != 0:
            continue

        frame_adjusted = adjust_brightness_contrast(frame, brightness_adjustment, contrast_adjustment)

        if len(frame_adjusted.shape) == 3:
            gray = cv2.cvtColor(frame_adjusted, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame_adjusted

        if image_size is None:
            image_size = gray.shape[::-1]

        detected, marker_corners, marker_ids, charuco_corners, charuco_ids = detect_and_interpolate_charuco(
            gray, board, dictionary, detector, charuco_detector, api_type
        )

        if detected:
            all_charuco_corners.append(charuco_corners)
            all_charuco_ids.append(charuco_ids)
            frames_used += 1
            if frames_used % 5 == 0:
                print(f"  Processed {frames_used} calibration frames...")

    print(f"Calibration using {frames_used} frames")

    if frames_used < CALIBRATION_FRAMES_NEEDED:
        return {
            "error": f"Insufficient calibration frames. Found {frames_used}, need {CALIBRATION_FRAMES_NEEDED}"}

    try:
        print("Running camera calibration...")

        # Try the new standard API (works for all OpenCV versions)
        retval, camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_camera_new_api(
            all_charuco_corners, all_charuco_ids, board, image_size
        )

        print(f"✓ Calibration converged: {retval}")
    except Exception as e:
        print(f"Calibration failed: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Calibration failed: {str(e)}"}

    if not retval:
        return {"error": "Calibration did not converge"}

    # Calculate reprojection error
    mean_error = 0
    errors_per_frame = []

    try:
        if hasattr(board, 'getChessboardCorners'):
            chessboard_corners = board.getChessboardCorners()
        elif hasattr(board, 'chessboardCorners'):
            chessboard_corners = board.chessboardCorners
        else:
            chessboard_corners = board.objPoints
    except:
        chessboard_corners = board.objPoints if hasattr(board, 'objPoints') else None

    if chessboard_corners is not None:
        for i in range(len(all_charuco_corners)):
            try:
                reprojected_points, _ = cv2.projectPoints(
                    chessboard_corners[all_charuco_ids[i]],
                    rvecs[i], tvecs[i],
                    camera_matrix, dist_coeffs
                )
                error = cv2.norm(all_charuco_corners[i], reprojected_points, cv2.NORM_L2) / len(reprojected_points)
                mean_error += error
                errors_per_frame.append(float(error))
            except:
                continue

    if errors_per_frame:
        mean_error /= len(errors_per_frame)
    else:
        mean_error = 0

    fov = calculate_fov(camera_matrix, image_size)
    calibration_quality = assess_calibration_quality(mean_error, frames_used)

    # Extend distortion coefficients to 8 values (PhotonVision uses 8-coefficient model)
    dist_coeffs_full = dist_coeffs.flatten().tolist()
    while len(dist_coeffs_full) < 8:
        dist_coeffs_full.append(0.0)
    dist_coeffs_full = dist_coeffs_full[:8]  # Take first 8

    # Create PhotonVision format (exact match to their JSON structure)
    photon_calibration = {
        "resolution": {
            "width": float(image_size[0]),
            "height": float(image_size[1])
        },
        "cameraIntrinsics": {
            "rows": 3,
            "cols": 3,
            "type": 6,  # CV_64F (64-bit float)
            "data": camera_matrix.flatten().tolist()
        },
        "distCoeffs": {
            "rows": 1,
            "cols": 8,
            "type": 6,  # CV_64F
            "data": dist_coeffs_full
        },
        "observations": [],  # Empty - PhotonVision doesn't require this
        "calobjectWarp": [0.0, 0.0],  # 2 values for warp
        "calobjectSize": {
            "width": 7.0,
            "height": 7.0
        },
        "calobjectSpacing": 0.0254,  # 25.4mm = 1 inch spacing
        "lensmodel": "LENSMODEL_OPENCV"
    }

    return {
        "success": True,
        "camera_model": "OV2311",
        "calibration_quality": calibration_quality,
        "reprojection_error": {
            "mean": float(mean_error),
            "per_frame": errors_per_frame,
            "max": float(max(errors_per_frame)) if errors_per_frame else 0,
            "min": float(min(errors_per_frame)) if errors_per_frame else 0,
        },
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": dist_coeffs.flatten().tolist(),
        "field_of_view": fov,
        "calibration_stats": {
            "frames_used": frames_used,
            "frames_processed": len(frames),
            "board_detection_rate": frames_used / max(1, len(frames) / FRAME_SKIP),
            "coverage_positions": len(detection_positions)
        },
        "image_size": {
            "width": int(image_size[0]),
            "height": int(image_size[1])
        },
        "intrinsic_parameters": {
            "focal_length_x": float(camera_matrix[0, 0]),
            "focal_length_y": float(camera_matrix[1, 1]),
            "principal_point_x": float(camera_matrix[0, 2]),
            "principal_point_y": float(camera_matrix[1, 2]),
            "skew": float(camera_matrix[0, 1])
        },
        "photonvision": photon_calibration
    }


def assess_calibration_quality(reprojection_error, frames_used):
    """Assess calibration quality"""
    if reprojection_error < 0.1:
        quality = "Excellent"
        score = 5
    elif reprojection_error < 0.3:
        quality = "Good"
        score = 4
    elif reprojection_error < 0.5:
        quality = "Acceptable"
        score = 3
    elif reprojection_error < 1.0:
        quality = "Poor"
        score = 2
    else:
        quality = "Unreliable"
        score = 1

    if frames_used < 20:
        score = max(1, score - 1)
    elif frames_used > 50:
        score = min(5, score + 1)

    return {
        "level": quality,
        "score": score,
        "description": f"Reprojection error: {reprojection_error:.3f} pixels, using {frames_used} frames"
    }


def calibrate_from_video(video_path):
    """Calibrate from video file"""
    global detected_frames

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video file"}

    board, dictionary, detector, charuco_detector, api_type = create_charuco_board_and_detector()
    frames = []
    detected_frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        detected, frame_with_detection, corners, ids, center = detect_charuco_board(
            frame, board, dictionary, detector, charuco_detector, api_type
        )

        if detected and corners is not None and ids is not None:
            detected_frames.append({'frame': frame, 'corners': corners, 'ids': ids})
            if frame_count % FRAME_SKIP == 0:
                frames.append(frame)

        if len(frames) > 500:
            break

    cap.release()

    if len(frames) < CALIBRATION_FRAMES_NEEDED:
        return {"error": f"Not enough frames. Found {len(detected_frames)}, need {CALIBRATION_FRAMES_NEEDED}"}

    return calibrate_from_frames(frames, board, dictionary, detector, charuco_detector, api_type)


def generate_frames():
    """Generate video frames with detection overlay"""
    global capture_running, camera_cap, last_detection_image, detection_count, frame_count, captured_frames, detected_frames, detection_positions

    board, dictionary, detector, charuco_detector, api_type = create_charuco_board_and_detector()
    consecutive_failures = 0
    max_failures = 10

    print(f"Starting frame generation (API: {api_type})...")

    while capture_running:
        try:
            with camera_lock:
                if camera_cap is None or not camera_cap.isOpened():
                    time.sleep(0.5)
                    continue

                ret, frame = camera_cap.read()

            if not ret or frame is None:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    print("Too many failures, stopping...")
                    break
                time.sleep(0.1)
                continue

            consecutive_failures = 0
            frame_count += 1
            frame_shape = frame.shape

            # Detect board
            detected, frame_with_detection, corners, ids, center = detect_charuco_board(
                frame, board, dictionary, detector, charuco_detector, api_type
            )

            if detected:
                detection_count += 1
                if center is not None:
                    detection_positions.append(center)

                if frame_count % FRAME_SKIP == 0:
                    captured_frames.append(frame.copy())
                    if corners is not None and ids is not None:
                        detected_frames.append({'frame': frame.copy(), 'corners': corners, 'ids': ids})

            # Draw coverage map
            frame_with_detection = draw_detection_coverage(frame_with_detection, detection_positions, frame_shape)

            # Add overlays
            h = frame_with_detection.shape[0]
            cv2.putText(frame_with_detection, f"OV2311 - Frames: {frame_count}", (10, h - 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame_with_detection, f"Detections: {detection_count}", (10, h - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame_with_detection, f"Brightness: {brightness_adjustment}", (10, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            color = (0, 255, 0) if len(captured_frames) >= CALIBRATION_FRAMES_NEEDED else (255, 255, 255)
            cv2.putText(frame_with_detection, f"Captured: {len(captured_frames)}", (10, h - 0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            ret, buffer = cv2.imencode('.jpg', frame_with_detection, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            last_detection_image = frame_bytes

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except Exception as e:
            print(f"Error in generate_frames: {e}")
            time.sleep(0.1)

    print("Frame generation stopped")


def start_capture(resolution):
    """Start capturing from OV2311"""
    global capture_running, captured_frames, detected_frames, camera_cap
    global detection_count, frame_count, last_detection_image, detection_positions

    print(f"\n{'=' * 60}")
    print(f"Starting OV2311 Camera")
    print(f"Resolution: {resolution}")
    print(f"{'=' * 60}")

    if capture_running:
        stop_capture()
        time.sleep(1.0)

    capture_running = False
    captured_frames = []
    detected_frames = []
    detection_positions = []
    detection_count = 0
    frame_count = 0
    last_detection_image = None

    try:
        with camera_lock:
            camera_id = 0
            camera_cap = None

            # Try V4L2
            print("Trying V4L2...")
            camera_cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
            if camera_cap.isOpened():
                print("✓ Opened with V4L2")
            else:
                camera_cap.release()
                camera_cap = None

            # Try DirectShow
            if camera_cap is None:
                print("Trying DirectShow...")
                camera_cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
                if camera_cap.isOpened():
                    print("✓ Opened with DirectShow")
                else:
                    camera_cap.release()
                    camera_cap = None

            # Try default
            if camera_cap is None:
                print("Trying default...")
                camera_cap = cv2.VideoCapture(camera_id)
                if camera_cap.isOpened():
                    print("✓ Opened with default")
                else:
                    return False

            if camera_cap is None or not camera_cap.isOpened():
                print("❌ Failed to open camera")
                return False

            # Settings
            width, height = resolution
            camera_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            camera_cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            camera_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            camera_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            camera_cap.set(cv2.CAP_PROP_FPS, 30)

            time.sleep(1.0)

            # Warmup
            for i in range(10):
                ret, frame = camera_cap.read()
                if ret and frame is not None and i == 0:
                    print(f"  Frame shape: {frame.shape}")
                time.sleep(0.05)

            # Test
            ret, test_frame = camera_cap.read()
            if not ret or test_frame is None:
                print("❌ Failed to read test frame")
                camera_cap.release()
                camera_cap = None
                return False

            print(f"✓ OV2311 Ready! {test_frame.shape}")
            print(f"{'=' * 60}\n")

        capture_running = True
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        capture_running = False
        with camera_lock:
            if camera_cap:
                camera_cap.release()
                camera_cap = None
        return False


def stop_capture():
    """Stop capture"""
    global capture_running, camera_cap

    print("Stopping capture...")
    capture_running = False
    time.sleep(0.5)

    with camera_lock:
        if camera_cap and camera_cap.isOpened():
            camera_cap.release()
            camera_cap = None
            print("✓ Camera released")

    return True


@app.route('/')
def index():
    """Main interface"""
    resolution_options = ""
    for width, height in RESOLUTIONS:
        selected = 'selected' if (width, height) == (800, 600) else ''
        resolution_options += f'<option value="{width},{height}" {selected}>{width}×{height}</option>'

    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>OV2311 Calibration</title>
    <style>
        body { font-family: Arial; max-width: 1400px; margin: 0 auto; padding: 20px; background: #f0f2f5; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; text-align: center; }
        .container { display: flex; gap: 30px; }
        .left-panel { flex: 1; }
        .right-panel { flex: 2; }
        .panel { background: white; padding: 25px; border-radius: 10px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); margin-bottom: 20px; }
        .controls { display: flex; flex-direction: column; gap: 15px; margin: 20px 0; }
        button { padding: 12px 24px; border: none; border-radius: 6px; cursor: pointer; font-size: 16px; font-weight: 600; transition: all 0.3s; display: flex; align-items: center; justify-content: center; gap: 8px; }
        button.primary { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
        button.danger { background: #dc3545; color: white; }
        button.success { background: #28a745; color: white; }
        button.secondary { background: #6c757d; color: white; }
        button:disabled { background: #ccc; cursor: not-allowed; opacity: 0.6; }
        .video-container { background: #1a1a1a; border-radius: 8px; overflow: hidden; margin: 20px 0; min-height: 400px; display: flex; align-items: center; justify-content: center; }
        .video-container img { max-width: 100%; max-height: 600px; object-fit: contain; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
        .stat-card { background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; border-left: 4px solid #667eea; }
        .stat-value { font-size: 32px; font-weight: bold; color: #333; margin: 10px 0; }
        .stat-label { font-size: 14px; color: #666; text-transform: uppercase; }
        .status { padding: 15px; margin: 15px 0; border-radius: 8px; font-weight: 500; }
        .status.success { background: #d4edda; color: #155724; border-left: 4px solid #28a745; }
        .status.error { background: #f8d7da; color: #721c24; border-left: 4px solid #dc3545; }
        .status.info { background: #d1ecf1; color: #0c5460; border-left: 4px solid #17a2b8; }
        select { padding: 12px; border: 2px solid #e1e5e9; border-radius: 6px; font-size: 16px; width: 100%; background: white; }
        .slider-container { margin: 15px 0; }
        .slider-container label { display: block; margin-bottom: 5px; font-weight: 600; }
        .slider { width: 100%; height: 8px; border-radius: 5px; background: #d3d3d3; outline: none; }
        .slider::-webkit-slider-thumb { width: 20px; height: 20px; border-radius: 50%; background: #667eea; cursor: pointer; }
        .slider-value { display: inline-block; padding: 4px 12px; background: #f8f9fa; border-radius: 4px; font-weight: 600; margin-left: 10px; }
        .progress-bar { height: 20px; background: #e9ecef; border-radius: 10px; overflow: hidden; }
        .progress-fill { height: 100%; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); width: 0%; transition: width 0.5s ease; }
        .quality-excellent { background: linear-gradient(135deg, #28a745 0%, #20c997 100%); }
        .quality-good { background: linear-gradient(135deg, #17a2b8 0%, #20c997 100%); }
        .quality-acceptable { background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%); }
        .quality-poor { background: linear-gradient(135deg, #fd7e14 0%, #dc3545 100%); }
        .tab { display: none; }
        .tab.active { display: block; }
        .tab-nav { display: flex; margin-bottom: 20px; background: #f8f9fa; border-radius: 8px; padding: 5px; }
        .tab-btn { flex: 1; padding: 12px 20px; background: none; border: none; border-radius: 6px; cursor: pointer; font-weight: 600; color: #666; }
        .tab-btn.active { background: white; color: #667eea; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .info-box { padding: 15px; background: #fff3cd; border-left: 4px solid #ffc107; border-radius: 4px; margin: 15px 0; }
    </style>
</head>
<body>
    <div class="header">
        <h1 style="margin: 0;">📷 OV2311 Camera Calibration</h1>
        <p style="margin: 10px 0 0;">OpenCV 4.13+ | Detection Tracking | Coverage Map</p>
    </div>

    <div class="tab-nav">
        <button class="tab-btn active" onclick="switchTab('capture')">🎥 Live</button>
        <button class="tab-btn" onclick="switchTab('upload')">📁 Upload</button>
        <button class="tab-btn" onclick="switchTab('results')" id="resultsTabBtn" style="display:none;">📊 Results</button>
    </div>

    <div class="container">
        <div class="left-panel">
            <div id="capture" class="tab active panel">
                <h2>Camera Settings</h2>

                <div class="controls">
                    <label>Resolution:</label>
                    <select id="resolutionSelect">''' + resolution_options + '''</select>
                </div>

                <div class="slider-container">
                    <label>Brightness: <span class="slider-value" id="brightnessValue">0</span></label>
                    <input type="range" min="-100" max="100" value="0" class="slider" id="brightnessSlider">
                </div>

                <div class="slider-container">
                    <label>Contrast: <span class="slider-value" id="contrastValue">1.0</span></label>
                    <input type="range" min="50" max="200" value="100" class="slider" id="contrastSlider">
                </div>

                <div class="info-box">
                    💡 <strong>Yellow dots</strong> on video show where board was detected. Cover different areas for better calibration!
                </div>

                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-label">Frames</div>
                        <div class="stat-value" id="frameCount">0</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Detections</div>
                        <div class="stat-value" id="detectionCount">0</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Time</div>
                        <div class="stat-value" id="captureTime">0s</div>
                    </div>
                </div>

                <div style="margin: 20px 0;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span>Progress: <span id="progressText">0%</span></span>
                        <span><span id="detectedFramesCount">0</span>/15 frames</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" id="progressFill"></div>
                    </div>
                </div>

                <div class="controls">
                    <button class="primary" onclick="startCapture()" id="startBtn">▶ Start</button>
                    <button class="danger" onclick="stopCapture()" id="stopBtn" disabled>⏹ Stop</button>
                    <button class="success" onclick="calibrateFromCapture()" id="calibrateBtn" disabled>🔧 Calibrate</button>
                </div>

                <div class="status info" id="captureStatus">Ready to calibrate OV2311</div>
            </div>

            <div id="upload" class="tab panel">
                <h2>Upload Video</h2>
                <div class="controls">
                    <input type="file" id="videoFile" accept="video/*">
                    <button class="primary" onclick="uploadVideo()">📤 Upload & Calibrate</button>
                </div>
                <div id="uploadStatus" class="status" style="display:none;"></div>
            </div>

            <div id="results" class="tab panel">
                <h2>Results</h2>
                <div id="resultsContent"><p style="text-align: center; color: #666; padding: 40px;">Results will appear here</p></div>
            </div>
        </div>

        <div class="right-panel">
            <div class="panel">
                <h2>Live Feed - Coverage Map</h2>
                <div class="video-container">
                    <img id="videoFeed" src="/video_feed" style="display:none;">
                    <div id="noFeed" style="color: #999; text-align: center; padding: 40px;">
                        <div style="font-size: 48px; margin-bottom: 20px;">📷</div>
                        <div>OV2311 feed will appear here</div>
                        <div style="margin-top: 15px; font-size: 14px;">Yellow dots = detected positions</div>
                    </div>
                </div>
                <div style="padding: 15px; background: #f8f9fa; border-radius: 8px; margin-top: 20px;">
                    <h3 style="margin-top: 0;">💡 Tips</h3>
                    <ul style="margin-bottom: 0; font-size: 14px;">
                        <li><strong>Yellow dots</strong> show detection coverage</li>
                        <li><strong>Magenta circle</strong> = current board position</li>
                        <li>Move board to cover all areas (corners, edges, center)</li>
                        <li>Different distances and angles improve calibration</li>
                        <li>Adjust brightness if not detecting</li>
                    </ul>
                </div>
            </div>
        </div>
    </div>

    <script>
        let captureInterval = null;
        let startTime = null;
        let isCapturing = false;

        document.getElementById('brightnessSlider').oninput = function() {
            const val = this.value;
            document.getElementById('brightnessValue').textContent = val;
            if (isCapturing) {
                fetch('/set_brightness', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({brightness: parseInt(val), contrast: parseFloat(document.getElementById('contrastSlider').value) / 100})
                });
            }
        };

        document.getElementById('contrastSlider').oninput = function() {
            const val = (this.value / 100).toFixed(1);
            document.getElementById('contrastValue').textContent = val;
            if (isCapturing) {
                fetch('/set_brightness', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({brightness: parseInt(document.getElementById('brightnessSlider').value), contrast: parseFloat(val)})
                });
            }
        };

        function switchTab(tabName) {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.getElementById(tabName).classList.add('active');
        }

        async function startCapture() {
            const res = document.getElementById('resolutionSelect').value.split(',');
            showStatus('captureStatus', 'Starting OV2311...', 'info');
            document.getElementById('startBtn').disabled = true;

            try {
                const r = await fetch('/start_capture', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({width: parseInt(res[0]), height: parseInt(res[1])})
                });
                const data = await r.json();
                if (r.ok && data.status === 'capture_started') {
                    isCapturing = true;
                    startTime = Date.now();
                    document.getElementById('startBtn').disabled = true;
                    document.getElementById('stopBtn').disabled = false;
                    document.getElementById('noFeed').style.display = 'none';
                    document.getElementById('videoFeed').style.display = 'block';
                    document.getElementById('videoFeed').src = '/video_feed?' + new Date().getTime();
                    captureInterval = setInterval(updateUI, 500);
                    showStatus('captureStatus', 'OV2311 active! Move board around - watch yellow dots!', 'success');
                } else {
                    document.getElementById('startBtn').disabled = false;
                    showStatus('captureStatus', 'Failed: ' + (data.error || 'Unknown'), 'error');
                }
            } catch (e) {
                document.getElementById('startBtn').disabled = false;
                showStatus('captureStatus', 'Error: ' + e.message, 'error');
            }
        }

        async function updateUI() {
            if (!isCapturing) return;
            try {
                const r = await fetch('/get_capture_stats');
                if (r.ok) {
                    const s = await r.json();
                    document.getElementById('frameCount').textContent = s.frame_count;
                    document.getElementById('detectionCount').textContent = s.detection_count;
                    document.getElementById('detectedFramesCount').textContent = s.captured_frames;
                    const p = Math.min(100, (s.captured_frames / 15) * 100);
                    document.getElementById('progressFill').style.width = p + '%';
                    document.getElementById('progressText').textContent = Math.round(p) + '%';
                    if (s.captured_frames >= 15) document.getElementById('calibrateBtn').disabled = false;
                    if (startTime) {
                        const t = Math.floor((Date.now() - startTime) / 1000);
                        document.getElementById('captureTime').textContent = t + 's';
                    }
                }
            } catch (e) {}
        }

        async function stopCapture() {
            if (captureInterval) clearInterval(captureInterval);
            isCapturing = false;
            try {
                const r = await fetch('/stop_capture', {method: 'POST'});
                if (r.ok) {
                    const d = await r.json();
                    document.getElementById('startBtn').disabled = false;
                    document.getElementById('stopBtn').disabled = true;
                    document.getElementById('videoFeed').style.display = 'none';
                    document.getElementById('noFeed').style.display = 'block';
                    showStatus('captureStatus', `Stopped. ${d.captured_frames} frames`, 'success');
                    if (d.captured_frames >= 15) document.getElementById('calibrateBtn').disabled = false;
                }
            } catch (e) {}
        }

        async function calibrateFromCapture() {
            showStatus('captureStatus', 'Calibrating... This may take a few seconds.', 'info');
            document.getElementById('calibrateBtn').disabled = true;
            try {
                const r = await fetch('/calibrate_from_capture', {method: 'POST'});
                const res = await r.json();
                if (r.ok) {
                    showResultsTab();
                    displayResults(res);
                    showStatus('captureStatus', 'Complete!', 'success');
                } else {
                    showStatus('captureStatus', 'Failed: ' + res.error, 'error');
                    document.getElementById('calibrateBtn').disabled = false;
                }
            } catch (e) {
                showStatus('captureStatus', 'Error: ' + e.message, 'error');
                document.getElementById('calibrateBtn').disabled = false;
            }
        }

        async function uploadVideo() {
            const f = document.getElementById('videoFile').files[0];
            if (!f) { showStatus('uploadStatus', 'Select file', 'error'); return; }
            const fd = new FormData();
            fd.append('video', f);
            showStatus('uploadStatus', 'Processing...', 'info');
            try {
                const r = await fetch('/calibrate', {method: 'POST', body: fd});
                const res = await r.json();
                if (r.ok) {
                    showResultsTab();
                    displayResults(res);
                    showStatus('uploadStatus', 'Complete!', 'success');
                } else {
                    showStatus('uploadStatus', 'Error: ' + res.error, 'error');
                }
            } catch (e) {
                showStatus('uploadStatus', 'Failed: ' + e.message, 'error');
            }
        }

        function showResultsTab() {
            document.getElementById('resultsTabBtn').style.display = 'block';
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.getElementById('resultsTabBtn').classList.add('active');
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.getElementById('results').classList.add('active');
        }

        function displayResults(r) {
            const div = document.getElementById('resultsContent');
            if (!r.success) {
                div.innerHTML = `<div class="status error"><h3>Failed</h3><p>${r.error}</p></div>`;
                return;
            }
            const qc = 'quality-' + r.calibration_quality.level.toLowerCase();
            div.innerHTML = `
                <div style="text-align: center; margin-bottom: 30px;">
                    <div style="display: inline-block; padding: 8px 20px; border-radius: 20px; color: white; font-weight: bold; font-size: 18px;" class="${qc}">
                        ${r.calibration_quality.level} - ${r.calibration_quality.score}/5
                    </div>
                    <p>${r.calibration_quality.description}</p>
                </div>
                <div class="stats-grid">
                    <div class="stat-card"><div class="stat-label">Error</div><div class="stat-value">${r.reprojection_error.mean.toFixed(3)} px</div></div>
                    <div class="stat-card"><div class="stat-label">H-FOV</div><div class="stat-value">${r.field_of_view.horizontal.toFixed(1)}°</div></div>
                    <div class="stat-card"><div class="stat-label">V-FOV</div><div class="stat-value">${r.field_of_view.vertical.toFixed(1)}°</div></div>
                    <div class="stat-card"><div class="stat-label">Frames</div><div class="stat-value">${r.calibration_stats.frames_used}</div></div>
                </div>
                <div style="margin: 30px 0; padding-top: 20px; border-top: 2px solid #eee;">
                    <h3>Parameters</h3>
                    <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; font-family: monospace; font-size: 14px;">
                        <strong>Camera:</strong> OV2311<br>
                        <strong>Resolution:</strong> ${r.image_size.width}×${r.image_size.height}<br>
                        <strong>Coverage:</strong> ${r.calibration_stats.coverage_positions} positions<br>
                        <strong>Focal (fx,fy):</strong> ${r.intrinsic_parameters.focal_length_x.toFixed(2)}, ${r.intrinsic_parameters.focal_length_y.toFixed(2)}<br>
                        <strong>Center (cx,cy):</strong> ${r.intrinsic_parameters.principal_point_x.toFixed(2)}, ${r.intrinsic_parameters.principal_point_y.toFixed(2)}<br>
                        <strong>FOV:</strong> ${r.field_of_view.horizontal.toFixed(1)}° H × ${r.field_of_view.vertical.toFixed(1)}° V
                    </div>
                </div>

                <div class="info-box" style="background: #e7f3ff; border-left-color: #2196F3; margin-top: 20px;">
                    <strong>📸 PhotonVision Import Instructions:</strong><br>
                    1. Click "Download PhotonVision" button below<br>
                    2. In PhotonVision web UI: Go to <strong>Settings → Cameras</strong><br>
                    3. Select your OV2311 camera<br>
                    4. Click <strong>"Import Calibration"</strong><br>
                    5. Upload the downloaded JSON file<br>
                    <br>
                    ⚠️ <strong>Important:</strong> PhotonVision camera resolution must be set to <strong>${r.image_size.width}×${r.image_size.height}</strong>
                </div>

                <div style="display: flex; gap: 10px; margin-top: 20px;">
                    <button class="secondary" onclick="downloadResults()">📥 Download Full JSON</button>
                    <button class="primary" onclick="downloadPhotonVision()">📸 Download PhotonVision</button>
                </div>

                <div style="margin-top: 30px; padding-top: 20px; border-top: 2px solid #eee;">
                    <h3>PhotonVision Config</h3>
                    <button class="secondary" onclick="togglePhotonPreview()" style="margin-bottom: 10px;">
                        <span id="toggleText">▼ Show</span> PhotonVision Format
                    </button>
                    <pre id="photonPreview" style="display: none; background: #1a1a1a; color: #0f0; padding: 20px; border-radius: 8px; overflow: auto; max-height: 400px; font-size: 12px;">
${JSON.stringify(r.photonvision, null, 2)}
                    </pre>
                </div>
            `;
            window.lastResult = r;
        }

        function togglePhotonPreview() {
            const preview = document.getElementById('photonPreview');
            const toggleText = document.getElementById('toggleText');
            if (preview.style.display === 'none') {
                preview.style.display = 'block';
                toggleText.textContent = '▲ Hide';
            } else {
                preview.style.display = 'none';
                toggleText.textContent = '▼ Show';
            }
        }

        function downloadResults() {
            if (!window.lastResult) return;
            const blob = new Blob([JSON.stringify(window.lastResult, null, 2)], {type: 'application/json'});
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = `ov2311_calibration_full_${new Date().toISOString().slice(0,19).replace(/:/g,'-')}.json`;
            a.click();
        }

        function downloadPhotonVision() {
            if (!window.lastResult || !window.lastResult.photonvision) return;
            const blob = new Blob([JSON.stringify(window.lastResult.photonvision, null, 2)], {type: 'application/json'});
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            const res = window.lastResult.image_size;
            a.download = `photonvision-OV2311-${res.width}x${res.height}.json`;
            a.click();

            // Show success message
            alert('✓ PhotonVision calibration downloaded!\\n\\nImport this file in PhotonVision:\\nSettings → Cameras → Import Calibration');
        }

        function showStatus(id, msg, type) {
            const el = document.getElementById(id);
            el.textContent = msg;
            el.className = 'status ' + type;
            el.style.display = 'block';
        }
    </script>
</body>
</html>
''')


@app.route('/start_capture', methods=['POST'])
def start_capture_endpoint():
    data = request.json
    width = data.get('width', 800)
    height = data.get('height', 600)
    success = start_capture((width, height))
    if success:
        return jsonify({"status": "capture_started", "camera": "OV2311"})
    else:
        return jsonify({"status": "error", "error": "Failed to start OV2311"}), 500


@app.route('/stop_capture', methods=['POST'])
def stop_capture_endpoint():
    success = stop_capture()
    return jsonify(
        {"status": "capture_stopped", "captured_frames": len(captured_frames), "detection_count": detection_count})


@app.route('/set_brightness', methods=['POST'])
def set_brightness():
    global brightness_adjustment, contrast_adjustment
    data = request.json
    brightness_adjustment = data.get('brightness', 0)
    contrast_adjustment = data.get('contrast', 1.0)
    return jsonify({"status": "ok", "brightness": brightness_adjustment, "contrast": contrast_adjustment})


@app.route('/get_capture_stats', methods=['GET'])
def get_capture_stats():
    return jsonify({
        "frame_count": frame_count,
        "detection_count": detection_count,
        "captured_frames": len(captured_frames),
        "last_detection": last_detected
    })


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/calibrate_from_capture', methods=['POST'])
def calibrate_from_capture_endpoint():
    if len(captured_frames) < CALIBRATION_FRAMES_NEEDED:
        return jsonify({"error": f"Need {CALIBRATION_FRAMES_NEEDED} frames. Got {len(captured_frames)}."}), 400

    frames_to_use = captured_frames
    if len(captured_frames) > 200:
        step = len(captured_frames) // 100
        frames_to_use = captured_frames[::step]

    board, dictionary, detector, charuco_detector, api_type = create_charuco_board_and_detector()
    result = calibrate_from_frames(frames_to_use, board, dictionary, detector, charuco_detector, api_type)

    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


@app.route('/calibrate', methods=['POST'])
def calibrate_camera():
    if 'video' not in request.files:
        return jsonify({"error": "No video file"}), 400
    file = request.files['video']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        file.save(tmp.name)
        temp_path = tmp.name

    try:
        result = calibrate_from_video(temp_path)
        os.unlink(temp_path)
        if "error" in result:
            return jsonify(result), 400
        return jsonify(result)
    except Exception as e:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("=" * 70)
    print("OV2311 Calibration - Detection Tracking + Coverage Map")
    print("=" * 70)
    print(f"📦 OpenCV Version: {cv2.__version__}")

    try:
        board, dictionary, detector, charuco_detector, api_type = create_charuco_board_and_detector()
        print(f"✓ Detection method: {api_type}")
        print(f"✓ Calibration method: cv2.calibrateCamera()")
    except Exception as e:
        print(f"❌ Setup error: {e}")
        import traceback

        traceback.print_exc()

    print("=" * 70)
    print("🌐 Server: http://localhost:5000")
    print("📷 Features: Yellow dots = detection coverage map")
    print("=" * 70)

    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
