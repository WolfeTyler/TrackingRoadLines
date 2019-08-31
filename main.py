import matplotlib.pyplot as plt
import cv2
import os
from os.path import join, basename
from collections import deque
from lane_detection import color_frame_pipeline


if __name__ == '__main__':

    resize_h, resize_w = 540, 960

    verbose = True
    if verbose:
        plt.ion()
        figManager = plt.get_current_fig_manager()
        # figManager.window.showMaximized()

    # test image file path
    test_images_dir = join('data', 'test_images')

    # list of all test image file paths
    test_images = [join(test_images_dir, name) for name in os.listdir(test_images_dir)]

    # for loop to process each test image through color_frame_pipeline
    for test_img in test_images:

        print('Processing image: {}'.format(test_img))

        # path for out files created post-processing
        out_path = join('out', 'images', basename(test_img))

        # pre-processing of test image in opencv
        # cvtColor converts an image from one color space to another
        # imread creates a 3D array of pixel color info in BGR format
        # COLOR_BGR2RGB converts the image to RGB color for matplotlib
        in_image = cv2.cvtColor(cv2.imread(test_img, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        # color_frame_pipeline
        # takes as input a list of frames (RGB) and returns an image (RGB) with overlaid the inferred road lanes
        out_image = color_frame_pipeline([in_image], solid_lines=True)

        # converts image back to BGR pixel 3D array and saves output image to file path
        cv2.imwrite(out_path, cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR))

        if verbose:
            plt.imshow(out_image)
            plt.waitforbuttonpress()
    plt.close('all')

    # test on videos
    test_videos_dir = join('data', 'test_videos')
    test_videos = [join(test_videos_dir, name) for name in os.listdir(test_videos_dir)]

    for test_video in test_videos:

        print('Processing video: {}'.format(test_video))

        cap = cv2.VideoCapture(test_video)
        out = cv2.VideoWriter(join('out', 'videos', basename(test_video)),
                              fourcc=cv2.VideoWriter_fourcc(*'DIVX'),
                              fps=20.0, frameSize=(resize_w, resize_h))

        frame_buffer = deque(maxlen=10)
        while cap.isOpened():
            ret, color_frame = cap.read()
            if ret:
                color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
                color_frame = cv2.resize(color_frame, (resize_w, resize_h))
                frame_buffer.append(color_frame)
                blend_frame = color_frame_pipeline(frames=frame_buffer, solid_lines=True, temporal_smoothing=True)
                out.write(cv2.cvtColor(blend_frame, cv2.COLOR_RGB2BGR))
                cv2.imshow('blend', cv2.cvtColor(blend_frame, cv2.COLOR_RGB2BGR)), cv2.waitKey(1)
            else:
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()



