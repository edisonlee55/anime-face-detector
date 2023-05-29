import argparse
import functools
import pathlib

import cv2
import gradio as gr
import numpy as np
import PIL.Image
import torch

import anime_face_detector


def detect(
    img,
    face_score_threshold: float,
    landmark_score_threshold: float,
    detector: anime_face_detector.LandmarkDetector,
) -> PIL.Image.Image:
    if not img:
        return None

    image = cv2.imread(img)
    preds = detector(image)

    res = image.copy()
    for pred in preds:
        box = pred["bbox"]
        box, score = box[:4], box[4]
        if score < face_score_threshold:
            continue
        box = np.round(box).astype(int)

        lt = max(2, int(3 * (box[2:] - box[:2]).max() / 256))

        cv2.rectangle(res, tuple(box[:2]), tuple(box[2:]), (0, 255, 0), lt)

        pred_pts = pred["keypoints"]
        for *pt, score in pred_pts:
            if score < landmark_score_threshold:
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)
            pt = np.round(pt).astype(int)
            cv2.circle(res, tuple(pt), lt, color, cv2.FILLED)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)

    image_pil = PIL.Image.fromarray(res)
    return image_pil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--detector", type=str, default="yolov3", choices=["yolov3", "faster-rcnn"]
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", choices=["cuda:0", "cpu"]
    )
    parser.add_argument("--face-score-threshold", type=float, default=0.5)
    parser.add_argument("--landmark-score-threshold", type=float, default=0.3)
    parser.add_argument("--score-slider-step", type=float, default=0.05)
    parser.add_argument("--port", type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()

    sample_path = pathlib.Path("assets/input.jpg")
    if not sample_path.exists():
        torch.hub.download_url_to_file(
            "https://raw.githubusercontent.com/edisonlee55/hysts-anime-face-detector/main/assets/input.jpg",
            sample_path.as_posix(),
        )

    detector = anime_face_detector.create_detector(args.detector, device=args.device)
    func = functools.partial(detect, detector=detector)

    title = "edisonlee55/hysts-anime-face-detector"
    description = "Demo for edisonlee55/hysts-anime-face-detector. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below."
    article = "<a href='https://github.com/edisonlee55/hysts-anime-face-detector'>GitHub Repo</a>"

    gr.Interface(
        func,
        [
            gr.Image(type="filepath", label="Input"),
            gr.Slider(
                0,
                1,
                step=args.score_slider_step,
                value=args.face_score_threshold,
                label="Face Score Threshold",
            ),
            gr.Slider(
                0,
                1,
                step=args.score_slider_step,
                value=args.landmark_score_threshold,
                label="Landmark Score Threshold",
            ),
        ],
        gr.Image(type="pil", label="Output"),
        title=title,
        description=description,
        article=article,
        examples=[
            [
                sample_path.as_posix(),
                args.face_score_threshold,
                args.landmark_score_threshold,
            ],
        ],
        live=args.live,
    ).launch(
        debug=args.debug, share=args.share, server_port=args.port, enable_queue=True
    )


if __name__ == "__main__":
    main()
