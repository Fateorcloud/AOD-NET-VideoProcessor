import cv2
import numpy as np
import tensorflow as tf



IM_SIZE = (720, 1280)


video_path = "input_video.mp4"
vidcap = cv2.VideoCapture(video_path)
fps = vidcap.get(cv2.CAP_PROP_FPS)
success, image = vidcap.read()
count = 0

model = tf.keras.models.load_model('aod_net_fog_v2.h5')

# Initialize the video writer
output_video_path = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (IM_SIZE[1], IM_SIZE[0]))

while success:
    frame = tf.image.resize(image, size=IM_SIZE, antialias=True)
    frame = frame / 255.0
    img = tf.expand_dims(frame, 0)
    processed_frame = model(img)

    tensor = processed_frame*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    frame = cv2.cvtColor(tensor, cv2.COLOR_RGB2BGR)
    out.write(frame)
    
    success, image = vidcap.read()
    count += 1

# Release the video writer
out.release()

print("Video created successfully!")