import mrcnn.config
import mrcnn.model
import numpy as np
import cv2


COCO_MODEL_PATH = "./models/mask_rcnn_coco.h5"


class InferenceConfig(mrcnn.config.Config):
    NAME = "coco"
    NUM_CLASSES = 1 + 80  # COCO has 80 classes
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def load_coco_names(file_name):
    names = {}
    with open(file_name) as f:
        for id_, name in enumerate(f):
            names[id_] = name.split('\n')[0]
    return names


def segment_from_image(image_path):
    class_names = load_coco_names('coco.names')

    model = mrcnn.model.MaskRCNN(mode="inference", model_dir="./logs", config=InferenceConfig())
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    image = cv2.imread(image_path)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.detect([img], verbose=1)[0]
    for i in range(len(results['rois'])):
        left = results['rois'][i, 1]
        top = results['rois'][i, 0]
        right = results['rois'][i, 3]
        bottom = results['rois'][i, 2]
        text = class_names[results['class_ids'][i]] + str(round(results['scores'][i], 3))
        mask = results['masks'][..., i]
        np.random.seed(results['class_ids'][i])
        mask = np.stack([mask.astype('int') * np.ones(image.shape[:-1]) * np.random.randint(0, 255, 1),
                         mask.astype('int') * np.ones(image.shape[:-1]) * np.random.randint(0, 255, 1),
                         mask.astype('int') * np.ones(image.shape[:-1]) * np.random.randint(0, 255, 1)], axis=2).astype('uint8')
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 1)
        cv2.putText(image, text, (left, top), cv2.FONT_HERSHEY_DUPLEX, (right - left) / 300, (255, 255, 255), 1)
        image = cv2.addWeighted(image, 1, mask, 0.5, 0)
    cv2.imshow('result', image)
    cv2.waitKey()


def segment_from_video(video_path):
    class_names = load_coco_names('coco.names')

    model = mrcnn.model.MaskRCNN(mode="inference", model_dir="./logs", config=InferenceConfig())
    model.load_weights(COCO_MODEL_PATH, by_name=True)
    
    cap = cv2.VideoCapture(video_path)
    timer = 0
    results = None
    while True:
        timer += 1
        _, frame = cap.read()
        if timer % 30 == 0:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.detect([img], verbose=1)[0]
        if results is not None:
            for i in range(len(results['rois'])):
                left = results['rois'][i, 1]
                top = results['rois'][i, 0]
                right = results['rois'][i, 3]
                bottom = results['rois'][i, 2]
                text = class_names[results['class_ids'][i]] + str(round(results['scores'][i], 3))
                mask = results['masks'][..., i]
                np.random.seed(results['class_ids'][i])
                mask = np.stack([mask.astype('int') * np.ones(frame.shape[:-1]) * np.random.randint(0, 255, 1),
                                 mask.astype('int') * np.ones(frame.shape[:-1]) * np.random.randint(0, 255, 1),
                                 mask.astype('int') * np.ones(frame.shape[:-1]) * np.random.randint(0, 255, 1)], axis=2).astype('uint8')
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)
                cv2.putText(frame, text, (left, top), cv2.FONT_HERSHEY_DUPLEX, (right - left) / 300, (255, 255, 255), 1)
                frame = cv2.addWeighted(frame, 1, mask, 0.5, 0)
        cv2.imshow('result', frame)
        cv2.waitKey(1)


if __name__ == '__main__':
    segment_from_image('test.jpg')
    # segment_from_video(0)
