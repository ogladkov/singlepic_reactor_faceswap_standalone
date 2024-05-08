import copy
from typing import List, Tuple, Union

import cv2
import insightface
from insightface.app.face_analysis import Face, FaceAnalysis
import numpy as np
from omegaconf import DictConfig
from PIL import Image
import torch
from torchvision.transforms.functional import normalize

from .registry import ARCH_REGISTRY
from .r_archs import codeformer_arch
from .r_facelib.utils.face_restoration_helper import FaceRestoreHelper
from ..utils import img2tensor, tensor2img


class FaceSwapScript:

    def process(self,
        source_img: Image.Image,
        input_img: Image.Image,
        cfg: DictConfig
    ) -> Image.Image:
        self.source_img = source_img
        self.input_img = input_img
        self.cfg = cfg
        result = self.swap_face(
            source_img=self.source_img,
            target_img=self.input_img,
            swap_model=self.cfg.models.face_swap_model
        )

        return Image.fromarray(result)

    def swap_face(
            self,
            source_img: Union[Image.Image, None],
            target_img: Image.Image,
            swap_model: Union[str, None] = None,
    ) -> np.ndarray:
        result_image = target_img

        if swap_model is not None:

            if isinstance(source_img, str):  # source_img is a base64 string
                import base64, io

                if 'base64,' in source_img:  # check if the base64 string has a data URL scheme
                    # split the base64 string to get the actual base64 encoded image data
                    base64_data = source_img.split('base64,')[-1]
                    # decode base64 string to bytes
                    img_bytes = base64.b64decode(base64_data)
                else:
                    # if no data URL scheme, just decode
                    img_bytes = base64.b64decode(source_img)

                source_img = Image.open(io.BytesIO(img_bytes))

            target_img = cv2.cvtColor(np.array(target_img).astype(np.uint8), cv2.COLOR_RGB2BGR)

            if source_img is not None:
                source_img = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)

            source_faces = self.analyze_faces(source_img)
            target_faces = self.analyze_faces(target_img)

            if len(target_faces) == 0:
                return result_image

            face_swapper = insightface.model_zoo.get_model(
                self.cfg.models.face_swap_model.path,
                providers=self.cfg.providers
            )
            swapped = face_swapper.get(target_img, target_faces[0], source_faces[0])
            swapped = cv2.cvtColor(swapped, cv2.COLOR_BGR2RGB)

            # Face restoration
            restored = self.restore_face(
                input_image=swapped,
                face_restore_visibility=self.cfg.params.face_restore_visibility,
                codeformer_weight=self.cfg.params.codeformer_weight
            )

            return restored

    def analyze_faces(self, img_data: np.ndarray, det_size: Tuple=(640, 640)) -> List[Face]:
        face_analyser = copy.deepcopy(self.getAnalysisModel())
        face_analyser.prepare(ctx_id=0, det_size=det_size)

        return face_analyser.get(img_data)

    def getAnalysisModel(self) -> FaceAnalysis:
        analysis_model = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=self.cfg.providers,
            root=self.cfg.models.insightface.home
        )
        return analysis_model

    def restore_face(
            self,
            input_image: np.ndarray,
            face_restore_visibility: float,
            codeformer_weight: float,
    ) -> np.ndarray:
        result = input_image
        model_path = self.cfg.models.face_restoration_model.path
        device = 'cuda' if 'CUDAExecutionProvider' in self.cfg.providers else 'cpu'

        codeformer_net = ARCH_REGISTRY.get("CodeFormer")(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=["32", "64", "128", "256"],
        ).to(device)
        checkpoint = torch.load(model_path)["params_ema"]
        codeformer_net.load_state_dict(checkpoint)
        facerestore_model = codeformer_net.eval()
        image_np = np.expand_dims(result, 0)
        total_images = image_np.shape[0]

        for i in range(total_images):
            cur_image_np = image_np[i, :, :, ::-1]
            original_resolution = cur_image_np.shape[0:2]

            if facerestore_model is None:
                return result

            else:
                self.face_helper = FaceRestoreHelper(
                    1, face_size=512, crop_ratio=(1, 1), save_ext='png',
                    use_parse=True, device=device, cfg=self.cfg
                )

            self.face_helper.clean_all()
            self.face_helper.read_image(cur_image_np)
            self.face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
            self.face_helper.align_warp_face()

            cropped_face = self.face_helper.cropped_faces[0]
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

            with torch.no_grad():
                output = facerestore_model(cropped_face_t, w=codeformer_weight)[0]
                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))

            del output
            torch.cuda.empty_cache()

            if face_restore_visibility < 1:
                restored_face = cropped_face * (1 - face_restore_visibility) + restored_face * face_restore_visibility

            restored_face = restored_face.astype("uint8")
            self.face_helper.add_restored_face(restored_face)
            self.face_helper.get_inverse_affine(None)

            restored_img = self.face_helper.paste_faces_to_input_image()
            restored_img = restored_img[:, :, ::-1]

            if original_resolution != restored_img.shape[0:2]:
                restored_img = cv2.resize(
                    restored_img, (0, 0),
                    fx=original_resolution[1] / restored_img.shape[1],
                    fy=original_resolution[0] / restored_img.shape[0],
                    interpolation=cv2.INTER_LINEAR
                )

            self.face_helper.clean_all()

            return restored_img
