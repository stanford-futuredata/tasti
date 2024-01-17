import tasti
import torch
import torchvision
import time
import torch.utils.data as data_utils
from scipy.spatial import distance
from datasets import load_dataset, load_from_disk


COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


class Box:
    def __init__(self, box, object_name, confidence):
        self.box = box
        self.xmin = box[0]
        self.ymin = box[1]
        self.xmax = box[2]
        self.ymax = box[3]
        self.object_name = object_name
        self.confidence = confidence

    def __str__(self):
        return f'Box({self.xmin},{self.ymin},{self.xmax},{self.ymax},{self.object_name},{self.confidence})'

    def __repr__(self):
        return self.__str__()


class ImageDataSet(torch.utils.data.Dataset):

    def __init__(self, pictures, list_of_idxs=None, transform_fn=lambda x: x):
        if list_of_idxs is None:
            list_of_idxs = []
        self.pictures = pictures
        self.list_of_idxs = list_of_idxs
        self.transform_fn = transform_fn
        self.current_idx = 0
        self.lenght = len(pictures)
        self.preprocessing_time = 0

    def transform(self, image):
        start = time.time()
        if image.mode == "L" or image.mode == "RGBA":
            image = image.convert('RGB')
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        transformed = transforms(image)
        end = time.time()
        transform_time = end-start
        self.preprocessing_time += transform_time
        return transformed

    def __len__(self):
        return len(self.pictures)

    def __getitem__(self, idx):
        picture = self.transform(self.pictures[int(idx)]['image'])
        self.current_idx = idx
        if len(self.pictures) - 1 <= idx:
            print("Preprocessing Time: " + str(self.preprocessing_time))
        return picture


class ImageIndex(tasti.Index):
    def get_target_dnn(self):
        #model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, progress=True)
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=True)
        return model

    def get_embedding_dnn(self):
        model = torchvision.models.resnet152(pretrained=True, progress=True)
        return model

    def get_pretrained_embedding_dnn(self):
        model = torchvision.models.resnet152(pretrained=True, progress=True)
        model.fc = torch.nn.Identity()
        return model

    def get_target_dnn_dataset(self, train_or_test='train'):
        dataset = load_from_disk("~/../../mount-ssd/hendrik/imageNet/train")
        return ImageDataSet(dataset, transform_fn=transform_images)

    def get_embedding_dnn_dataset(self, train_or_test='test'):
        dataset = load_from_disk("~/../../mount-ssd/hendrik/imageNet/test")
        return ImageDataSet(dataset, transform_fn=transform_images)

    def target_dnn_callback(self, target_dnn_output):
        boxes = target_dnn_output[0]['boxes'].detach().cpu().numpy()
        confidences = target_dnn_output[0]['scores'].detach().cpu().numpy()
        object_ids = target_dnn_output[0]['labels'].detach().cpu().numpy()
        label = []
        for i in range(len(boxes)):
            object_name = COCO_INSTANCE_CATEGORY_NAMES[object_ids[i]]
            if confidences[i] > 0.95 and object_name in ['car']:
                box = Box(boxes[i], object_ids[i], confidences[i])
                label.append(box)
        return label

    def is_close(self, label1, label2):
        objects = set()
        for obj in (label1 + label2):
            objects.add(obj.object_name)
        for current_obj in list(objects):
            label1_disjoint = [obj for obj in label1 if obj.object_name == current_obj]
            label2_disjoint = [obj for obj in label2 if obj.object_name == current_obj]
            is_redundant = night_street_is_close_helper(label1_disjoint, label2_disjoint)
            if not is_redundant:
                return False
        return True


'''
Defines our notion of 'closeness' as described in the paper for two labels for only one object type.
'''


def night_street_is_close_helper(label1, label2):
    if len(label1) != len(label2):
        return False
    counter = 0
    for obj1 in label1:
        xavg1 = (obj1.xmin + obj1.xmax) / 2.0
        yavg1 = (obj1.ymin + obj1.ymax) / 2.0
        coord1 = [xavg1, yavg1]
        expected_counter = counter + 1
        for obj2 in label2:
            xavg2 = (obj2.xmin + obj2.xmax) / 2.0
            yavg2 = (obj2.ymin + obj2.ymax) / 2.0
            coord2 = [xavg2, yavg2]
            if distance.euclidean(coord1, coord2) < 100:
                counter += 1
                break
        if expected_counter != counter:
            break
    return len(label1) == counter


def transform_images(frame):
    return frame


class PictureAggregateQuery(tasti.AggregateQuery):
    def score(self, target_dnn_output):
        return len(target_dnn_output)


class PictureLimitQuery(tasti.LimitQuery):
    def score(self, target_dnn_output):
        return len(target_dnn_output)


class PicturePrecisionQuery(tasti.SUPGPrecisionQuery):
    def score(self, target_dnn_output):
        return 1.0 if len(target_dnn_output) > 0 else 0.0


class PictureSUPGRecallQuery(tasti.SUPGRecallQuery):
    def score(self, target_dnn_output):
        return 1.0 if len(target_dnn_output) > 0 else 0.0


class PictureLHSPrecisionQuery(tasti.SUPGPrecisionQuery):
    def score(self, target_dnn_output):
        def proc_boxes(boxes):
            mid = 1750 / 2
            for box in boxes:
                x = (box.xmin + box.xmax) / 2.
                if x < mid:
                    return True
            return False

        return proc_boxes(target_dnn_output)


class PictureLHSRecallQuery(tasti.SUPGRecallQuery):
    def score(self, target_dnn_output):
        def proc_boxes(boxes):
            mid = 1750 / 2
            for box in boxes:
                x = (box.xmin + box.xmax) / 2.
                if x < mid:
                    return True
            return False

        return proc_boxes(target_dnn_output)


class PictureAveragePositionAggregateQuery(tasti.AggregateQuery):
    def score(self, target_dnn_output):
        def proc_boxes(boxes):
            avg = 0.
            if len(boxes) == 0:
                return 0.
            for box in boxes:
                x = (box.xmin + box.xmax) / 2.
                avg += x / 1750
            return avg / len(boxes)

        return proc_boxes(target_dnn_output)


class ImageConfig(tasti.IndexConfig):
    def __init__(self):
        super().__init__()
        self.do_mining = True
        self.do_training = True
        self.do_infer = True
        self.do_bucketting = True

        self.batch_size = 16
        self.nb_train = 10000
        self.train_margin = 1.0
        self.train_lr = 1e-4
        self.max_k = 5
        self.nb_buckets = 300
        self.nb_training_its = 50000


if __name__ == '__main__':
    config = ImageConfig()
    index = ImageIndex(config=config)
    index.init()

    # query = PictureAggregateQuery(index)
    # query.execute(err_tol=0.1, confidence=0.1)

    query = PictureLimitQuery(index)
    start = time.time()
    query.execute(want_to_find=1, nb_to_find=1)
    end = time.time()
    query_time = end-start
    print("Execution Time for 1 picture: " + str(query_time))

    start = time.time()
    query.execute(want_to_find=1, nb_to_find=16)
    end = time.time()
    query_time = end-start
    print("Execution Time for 16 pictures: " + str(query_time))

    start = time.time()
    query.execute(want_to_find=1, nb_to_find=64)
    end = time.time()
    query_time = end-start
    print("Execution Time for 64 pictures: " + str(query_time))

    start = time.time()
    query.execute(want_to_find=1, nb_to_find=128)
    end = time.time()
    query_time = end-start
    print("Execution Time for 128 pictures: " + str(query_time))

    start = time.time()
    query.execute(want_to_find=1, nb_to_find=256)
    end = time.time()
    query_time = end-start
    print("Execution Time for 256 pictures: " + str(query_time))