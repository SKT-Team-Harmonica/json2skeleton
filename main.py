from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse
import os
import cv2
import json
import numpy as np
from pathlib import Path
import requests

app = FastAPI()

# 요청 바디를 위한 Pydantic 모델 정의
class KeypointsRequest(BaseModel):
    json_file_path: str

@app.post("/visualize-keypoints/")
async def visualize_keypoints(request: KeypointsRequest):
    json_file_path = request.json_file_path

    # 저장 경로 설정
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    output_video_path = output_dir / 'keypoints_visualization_final_test.mp4'

    # 원격 URL에서 JSON 파일 다운로드
    if json_file_path.startswith("http://") or json_file_path.startswith("https://"):
        try:
            response = requests.get(json_file_path)
            response.raise_for_status()
            json_input_path = output_dir / 'downloaded_keypoints.json'
            with open(json_input_path, 'w') as json_file:
                json.dump(response.json(), json_file)
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=400, detail=f"Failed to download JSON file: {e}")
    else:
        json_input_path = Path(json_file_path)
        if not json_input_path.exists():
            raise HTTPException(status_code=400, detail="JSON file not found")

    # 비디오 저장 설정 (코덱, FPS, 해상도)
    frame_width = 640  # 원하는 출력 영상의 너비
    frame_height = 480  # 원하는 출력 영상의 높이
    fps = 30  # 초당 프레임 수
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (frame_width, frame_height))

    # JSON 파일 읽기
    try:
        with open(json_input_path, 'r') as json_file:
            animation_data = json.load(json_file)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file")

    # 프레임마다 시각화
    for frame_data in animation_data['frames']:
        # 검정색 배경 이미지 생성
        image = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        keypoints = frame_data['keypoints']

        for ki, keypoint in enumerate(keypoints):
            keypoints[ki][1] = 1.0 - keypoint[1]

        # 포즈, 왼손, 오른손 키포인트 분리
        pose_keypoints = keypoints[:15]
        left_hand_keypoints = keypoints[15:36]
        right_hand_keypoints = keypoints[36:]

        # 포즈 키포인트 시각화 (초록색)
        for i in range(len(pose_keypoints)):
            x = int(pose_keypoints[i][0] * frame_width)
            y = int(pose_keypoints[i][1] * frame_height)
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

        # 포즈 키포인트 연결
        pose_connections = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
                            (9, 10), (11, 12), (11, 13), (12, 14)]
        for connection in pose_connections:
            start_idx, end_idx = connection
            if start_idx < len(pose_keypoints) and end_idx < len(pose_keypoints):
                x1, y1 = int(pose_keypoints[start_idx][0] * frame_width), int(
                    pose_keypoints[start_idx][1] * frame_height)
                x2, y2 = int(pose_keypoints[end_idx][0] * frame_width), int(pose_keypoints[end_idx][1] * frame_height)
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 왼손 키포인트 시각화 (파란색)
        for i in range(len(left_hand_keypoints)):
            x = int(left_hand_keypoints[i][0] * frame_width)
            y = int(left_hand_keypoints[i][1] * frame_height)
            cv2.circle(image, (x, y), 5, (255, 0, 0), -1)

        # 왼손 키포인트 연결
        hand_connections = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                            (5, 9), (9, 10), (10, 11), (11, 12),
                            (9, 13), (13, 14), (14, 15), (15, 16),
                            (13, 17), (17, 18), (18, 19), (19, 20)]
        for connection in hand_connections:
            start_idx, end_idx = connection
            if start_idx < len(left_hand_keypoints) and end_idx < len(left_hand_keypoints):
                x1, y1 = int(left_hand_keypoints[start_idx][0] * frame_width), int(
                    left_hand_keypoints[start_idx][1] * frame_height)
                x2, y2 = int(left_hand_keypoints[end_idx][0] * frame_width), int(
                    left_hand_keypoints[end_idx][1] * frame_height)
                cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # 오른손 키포인트 시각화 (빨간색)
        for i in range(len(right_hand_keypoints)):
            x = int(right_hand_keypoints[i][0] * frame_width)
            y = int(right_hand_keypoints[i][1] * frame_height)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

        # 오른손 키포인트 연결
        for connection in hand_connections:
            start_idx, end_idx = connection
            if start_idx < len(right_hand_keypoints) and end_idx < len(right_hand_keypoints):
                x1, y1 = int(right_hand_keypoints[start_idx][0] * frame_width), int(
                    right_hand_keypoints[start_idx][1] * frame_height)
                x2, y2 = int(right_hand_keypoints[end_idx][0] * frame_width), int(
                    right_hand_keypoints[end_idx][1] * frame_height)
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.line(image, (int(pose_keypoints[14][0] * frame_width), int(pose_keypoints[14][1] * frame_height)),
                 (int(right_hand_keypoints[0][0] * frame_width), int(right_hand_keypoints[0][1] * frame_height)),
                 (0, 255, 0), 2)

        cv2.line(image, (int(pose_keypoints[13][0] * frame_width), int(pose_keypoints[13][1] * frame_height)),
                 (int(left_hand_keypoints[0][0] * frame_width), int(left_hand_keypoints[0][1] * frame_height)),
                 (0, 255, 0), 2)

        # 영상 파일에 이미지 추가
        out.write(image)

    # 비디오 저장 마무리
    out.release()
    cv2.destroyAllWindows()

    # 파일 응답
    return FileResponse(str(output_video_path), media_type='video/mp4', filename='keypoints_visualization_final_test.mp4')
