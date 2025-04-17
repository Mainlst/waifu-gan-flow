import gradio as gr
import imageio
import numpy as np
import onnx
import onnxruntime as rt
import time
from numpy.random import RandomState
from skimage import transform
import json
import os

# 追加：保存先フォルダ
os.makedirs("outputs/images", exist_ok=True)
os.makedirs("outputs/videos", exist_ok=True)

def get_inter(r1, r2):
    h_inter = max(min(r1[3], r2[3]) - max(r1[1], r2[1]), 0)
    w_inter = max(min(r1[2], r2[2]) - max(r1[0], r2[0]), 0)
    return h_inter * w_inter


def iou(r1, r2):
    s1 = (r1[2] - r1[0]) * (r1[3] - r1[1])
    s2 = (r2[2] - r2[0]) * (r2[3] - r2[1])
    i = get_inter(r1, r2)
    return i / (s1 + s2 - i)


def letterbox(im, new_shape=(640, 640), color=(0.5, 0.5, 0.5), stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape != new_unpad:  # resize
        im = transform.resize(im, (new_unpad[1], new_unpad[0]))
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    im_new = np.full((new_unpad[1] + top + bottom, new_unpad[0] + left + right, 3), color, dtype=np.float32)
    im_new[top:new_unpad[1] + top, left:new_unpad[0] + left] = im
    return im_new


def nms(pred, conf_thres, iou_thres, max_instance=20):  # pred (anchor_num, 5 + cls_num)
    nc = pred.shape[1] - 5
    candidates = [list() for x in range(nc)]
    for x in pred:
        if x[4] < conf_thres:
            continue
        cls = np.argmax(x[5:])
        p = x[4] * x[5 + cls]
        if conf_thres <= p:
            box = (x[0] - x[2] / 2, x[1] - x[3] / 2, x[0] + x[2] / 2, x[1] + x[3] / 2)  # xywh2xyxy
            candidates[cls].append([p, box])
    result = [list() for x in range(nc)]
    for i, candidate in enumerate(candidates):
        candidate = sorted(candidate, key=lambda a: a[0], reverse=True)
        candidate = candidate[:max_instance]
        for x in candidate:
            ok = True
            for r in result[i]:
                if iou(r[1], x[1]) > iou_thres:
                    ok = False
                    break
            if ok:
                result[i].append(x)

    return result


class Model:
    def __init__(self):
        self.img_avg = None
        self.detector = None
        self.encoder = None
        self.g_synthesis = None
        self.g_mapping = None
        self.w_avg = None
        self.detector_stride = None
        self.detector_imgsz = None
        self.detector_class_names = None
        self.load_models("./models/")

    def load_models(self, model_dir):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        g_mapping = onnx.load(model_dir + "g_mapping.onnx")
        w_avg = [x for x in g_mapping.graph.initializer if x.name == "w_avg"][0]
        w_avg = np.frombuffer(w_avg.raw_data, dtype=np.float32)[np.newaxis, :]
        w_avg = w_avg.repeat(16, axis=0)[np.newaxis, :]
        self.w_avg = w_avg
        self.g_mapping = rt.InferenceSession(model_dir + "g_mapping.onnx", providers=providers)
        self.g_synthesis = rt.InferenceSession(model_dir + "g_synthesis.onnx", providers=providers)
        self.encoder = rt.InferenceSession(model_dir + "fb_encoder.onnx", providers=providers)
        self.detector = rt.InferenceSession(model_dir + "waifu_dect.onnx", providers=providers)
        detector_meta = self.detector.get_modelmeta().custom_metadata_map
        self.detector_stride = int(detector_meta['stride'])
        self.detector_imgsz = 1088
        self.detector_class_names = eval(detector_meta['names'])

        self.img_avg = transform.resize(
    self.g_synthesis.run(None, {'w': w_avg, 'noise': np.array([1.0], dtype=np.float32)})[0][0].transpose(1, 2, 0),
                                        (256, 256)).transpose(2, 0, 1)[np.newaxis, :]

    def get_img(self, w):
        img = self.g_synthesis.run(None, {'w': w, 'noise': np.array([1.0], dtype=np.float32)})[0]
        return (img.transpose(0, 2, 3, 1) * 127.5 + 128).clip(0, 255).astype(np.uint8)[0]


    def get_w(self, z, psi):
        psi_input = np.asarray([psi, 1.0], dtype=np.float32)  # ONNX モデルに合った shape に整える
        return self.g_mapping.run(None, {'z': z, 'psi': psi_input})[0]



    def encode_img(self, img, iteration=5):
        target_img = transform.resize(((img / 255 - 0.5) / 0.5), (256, 256)).transpose(2, 0, 1)[np.newaxis, :].astype(
            np.float32)
        w = self.w_avg.copy()
        from_img = self.img_avg.copy()
        for i in range(iteration):
            dimg = np.concatenate([target_img, from_img], axis=1)
            dw = self.encoder.run(None, {'dimg': dimg})[0]
            w += dw
            from_img = transform.resize(self.g_synthesis.run(None, {'w': w})[0][0].transpose(1, 2, 0),
                                        (256, 256)).transpose(2, 0, 1)[np.newaxis, :]
        return w

    def detect(self, im0, conf_thres, iou_thres, detail=False):
        if im0 is None:
            return []
        img = letterbox((im0 / 255).astype(np.float32), (self.detector_imgsz, self.detector_imgsz),
                        stride=self.detector_stride)
        # Convert
        img = img.transpose(2, 0, 1)
        img = img[np.newaxis, :]
        pred = self.detector.run(None, {'images': img})[0][0]
        dets = nms(pred, conf_thres, iou_thres)
        imgs = []
        # Print results
        s = '%gx%g ' % img.shape[2:]  # print string
        for i, det in enumerate(dets):
            n = len(det)
            s += f"{n} {self.detector_class_names[i]}{'s' * (n > 1)}, "  # add to string
        if detail:
            print(s)
        waifu_rects = []
        head_rects = []
        body_rects = []

        for i, det in enumerate(dets):
            for x in det:
                # Rescale boxes from img_size to im0 size
                wr = im0.shape[1] / img.shape[3]
                hr = im0.shape[0] / img.shape[2]
                x[1] = (int(x[1][0] * wr), int(x[1][1] * hr),
                        int(x[1][2] * wr), int(x[1][3] * hr))
                if i == 0:
                    head_rects.append(x[1])
                elif i == 1:
                    body_rects.append(x[1])
                elif i == 2:
                    waifu_rects.append(x[1])
        for j, waifu_rect in enumerate(waifu_rects):
            msg = f'waifu {j + 1} '
            head_num = 0
            body_num = 0
            hr, br = None, None
            for r in head_rects:
                if get_inter(r, waifu_rect) / ((r[2] - r[0]) * (r[3] - r[1])) > 0.75:
                    hr = r
                    head_num += 1
            if head_num != 1:
                if detail:
                    print(msg + f'head num error: {head_num}')
                continue
            for r in body_rects:
                if get_inter(r, waifu_rect) / ((r[2] - r[0]) * (r[3] - r[1])) > 0.65:
                    br = r
                    body_num += 1
            if body_num != 1:
                if detail:
                    print(msg + f'body num error: {body_num}')
                continue
            bounds = (int(min(waifu_rect[0], hr[0], br[0])),
                      int(min(waifu_rect[1], hr[1], br[1])),
                      int(max(waifu_rect[2], hr[2], br[2])),
                      int(max(waifu_rect[3], hr[3], br[3])))
            if (bounds[2] - bounds[0]) / (bounds[3] - bounds[1]) > 0.7:
                if detail:
                    print(msg + "ratio out of limit")
                continue
            # 扩展边界
            expand_pixel = (bounds[3] - bounds[1]) // 20
            bounds = (max(bounds[0] - expand_pixel // 2, 0),
                      max(bounds[1] - expand_pixel, 0),
                      min(bounds[2] + expand_pixel // 2, im0.shape[1]),
                      min(bounds[3] + expand_pixel, im0.shape[0]),
                      )
            if bounds[3] - bounds[1] >= (bounds[2] - bounds[0]) * 2:  # 等高度剪裁
                cx = (bounds[2] + bounds[0]) // 2
                h = bounds[3] - bounds[1]
                w = h // 2
                w2 = w // 2
                l1 = max(cx - w2, 0)
                r1 = min(cx + w2, im0.shape[1])
                bounds = (l1, bounds[1], r1, bounds[3])
                temp_bound = (w2 - (cx - l1), 0, w2 + (r1 - cx), h)
            else:  # 等宽度剪裁
                cy = (bounds[3] + bounds[1]) // 2
                w = bounds[2] - bounds[0]
                h = w * 2
                h2 = h // 2
                tp1 = max(cy - h2, 0)
                b1 = min(cy + h2, im0.shape[0])
                bounds = (bounds[0], tp1, bounds[2], b1)
                temp_bound = (0, h2 - (cy - tp1), w, h2 + (b1 - cy))
            temp_img = np.full((h, w, 3), 255, dtype=np.uint8)
            temp_img[temp_bound[1]:temp_bound[3], temp_bound[0]:temp_bound[2]] = im0[bounds[1]:bounds[3],
                                                                                 bounds[0]:bounds[2]]
            temp_img = transform.resize(temp_img, (1024, 512), preserve_range=True).astype(np.uint8)
            imgs.append(temp_img)
        return imgs

    def gen_video(self, w1, w2, path, frame_num=10):
        video = imageio.get_writer(path, mode='I', fps=frame_num // 2, codec='libx264', bitrate='16M')
        lin = np.linspace(0, 1, frame_num)
        for i in range(0, frame_num):
            img = self.get_img(((1 - lin[i]) * w1) + (lin[i] * w2))
            video.append_data(img)
        video.close()


def gen_fn(method, seed, psi):
    # （既存コード）
    z = (
        RandomState(int(seed) + 2**31).randn(1, 1024) if method == 1
        else np.random.randn(1, 1024)
    )
    w = model.get_w(z.astype(np.float32), psi)
    img_out = model.get_img(w)

    # 追加：ファイル名を作成＆保存
    filename = f"outputs/images/seed_{seed}_psi_{psi:.2f}.png"
    imageio.imwrite(filename, img_out)

    # 戻り値はそのまま表示用の img_out
    return img_out, w, img_out


def encode_img_fn(img):
    if img is None:
        return "please upload a image", None, None, None, None
    imgs = model.detect(img, 0.2, 0.03)
    if len(imgs) == 0:
        return "failed to detect waifu", None, None, None, None
    w = model.encode_img(imgs[0])
    img_out = model.get_img(w)
    return "success", imgs[0], img_out, w, img_out


def gen_video_fn(w1, w2, frame):
    if w1 is None or w2 is None:
        return None
    # 出力ファイル名を作成
    video_path = f"outputs/videos/video_{int(time.time())}.mp4"
    model.gen_video(w1, w2, video_path, int(frame))
    return video_path  # Gradio に渡すのは保存したファイルパス


if __name__ == '__main__':
    model = Model()

    app = gr.Blocks()
    with app:
        gr.Markdown("# full-body anime\n\n"
                    "the model is not well, just use for fun.")
        with gr.Tabs():
            with gr.TabItem("generate image"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("generate image randomly or by seed")
                        gen_input1 = gr.Radio(label="method", choices=["random", "use seed"], type="index")
                        gen_input2 = gr.Number(value=1, label="seed")
                        gen_input3 = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.6, label="truncation psi")
                        with gr.Group():
                            gen_submit = gr.Button("Generate")
                    with gr.Column():
                        gen_output1 = gr.Image()
                        select_img_input_w1 = gr.Variable()
                        select_img_input_img1 = gr.Variable()

            with gr.TabItem("encode image"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("you'd better upload a standing full-body image")
                        encode_img_input = gr.Image()
                        with gr.Group():
                            encode_img_submit = gr.Button("Run")
                    with gr.Column():
                        encode_img_output1 = gr.Textbox(label="message")
                        with gr.Row():
                            encode_img_output2 = gr.Image(label="detected")
                            encode_img_output3 = gr.Image(label="encoded")
                        select_img_input_w2 = gr.Variable()
                        select_img_input_img2 = gr.Variable()

            with gr.TabItem("generate video"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("## generate video between 2 images")
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("please select image 1")
                                select_img1_dropdown = gr.Radio(label="source",
                                                                choices=["current generated image",
                                                                         "current encoded image"], type="index")
                                with gr.Group():
                                    select_img1_button = gr.Button("select")
                                select_img1_output_img = gr.Image(label="image 1")
                                select_img1_output_w = gr.Variable()
                            with gr.Column():
                                gr.Markdown("please select image 2")
                                select_img2_dropdown = gr.Radio(label="source",
                                                                choices=["current generated image",
                                                                         "current encoded image"], type="index")
                                with gr.Group():
                                    select_img2_button = gr.Button("select")
                                select_img2_output_img = gr.Image(label="image 2")
                                select_img2_output_w = gr.Variable()
                        generate_video_frame = gr.Slider(minimum=10, maximum=30, step=1, label="frame", value=15)
                        with gr.Group():
                            generate_video_button = gr.Button("Generate")
                    with gr.Column():
                        generate_video_output = gr.Video()

        gen_submit.click(gen_fn, [gen_input1, gen_input2, gen_input3],
                         [gen_output1, select_img_input_w1, select_img_input_img1])
        encode_img_submit.click(encode_img_fn, [encode_img_input],
                                [encode_img_output1, encode_img_output2, encode_img_output3, select_img_input_w2,
                                 select_img_input_img2])
        select_img1_button.click(lambda i, img1, img2, w1, w2: (img1, w1) if i == 0 else (img2, w2),
                                 [select_img1_dropdown, select_img_input_img1, select_img_input_img2,
                                  select_img_input_w1, select_img_input_w2],
                                 [select_img1_output_img, select_img1_output_w])
        select_img2_button.click(lambda i, img1, img2, w1, w2: (img1, w1) if i == 0 else (img2, w2),
                                 [select_img2_dropdown, select_img_input_img1, select_img_input_img2,
                                  select_img_input_w1, select_img_input_w2],
                                 [select_img2_output_img, select_img2_output_w])
        generate_video_button.click(gen_video_fn, [select_img1_output_w, select_img2_output_w, generate_video_frame],
                                    [generate_video_output])

    app.launch(share=True)

