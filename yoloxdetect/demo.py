from yoloxdetect.utils.downloads import attempt_download_from_hub
from yolox.data.datasets import COCO_CLASSES
from yolox.data.data_augment import preproc
from yolox.utils import postprocess, vis
import importlib
import torch
import cv2
import os


class YoloxDetector:
    def __init__(
        self,
        model_path: str,
        config_path: str,
        device: str = "cpu",
        hf_model: bool = False,
    ):

        self.device = device
        self.config_path = config_path
        self.classes = COCO_CLASSES
        self.conf = 0.3
        self.iou = 0.45
        self.show = False  
        self.save = True
        
        if self.save:
            self.save_path = 'output/result.jpg'
            
        if hf_model:
            self.model_path = attempt_download_from_hub(model_path)
        
        else:
            self.model_path = model_path
        
    
        self.load_model()


    def load_model(self):
        current_exp = importlib.import_module(self.config_path)
        exp = current_exp.Exp()
        
        model = exp.get_model()
        model.to(self.device)
        model.eval()
        ckpt = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(ckpt["model"])
        self.model = model


    def predict(self, image_path, image_size):    
        image = cv2.imread(image_path)    
        if image_size is not None:
            ratio = min(image_size / image.shape[0], image_size / image.shape[1])
            img, _ = preproc(image, input_size=(image_size, image_size))
            img = torch.from_numpy(img).to(self.device).unsqueeze(0).float()
        else:
            manuel_size = 640
            ratio = min(manuel_size / image.shape[0], manuel_size / image.shape[1])
            img, _ = preproc(image, input_size=(manuel_size, manuel_size))
            img = torch.from_numpy(img).to(self.device).unsqueeze(0).float()

        prediction_result = self.model(img)
        original_predictions = postprocess(
            prediction=prediction_result,
            num_classes= len(COCO_CLASSES),
            conf_thre=self.conf,
            nms_thre=self.iou)[0]
        
        output = original_predictions.cpu()
        bboxes = output[:, 0:4]
        bboxes /= ratio
        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(
            image,
            bboxes,
            scores,
            cls,
            self.conf,
            COCO_CLASSES,
        )
        if self.show:
            cv2.imshow("result", vis_res)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif self.save:
            save_dir = self.save_path[:self.save_path.rfind('/')]
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            cv2.imwrite(self.save_path, vis_res)
            return self.save_path
        
        else:
            return vis_res


if __name__ == "__main__":
    model = YoloxDetector(
        model_path = "kadirnar/yolox_s-v0.1.1",
        config_path = "configs.yolox_s",
        device = "cuda:0",
        hf_model=True,
    )

    image = "data/images/dog.jpg"
    
    model.predict(image, 640)