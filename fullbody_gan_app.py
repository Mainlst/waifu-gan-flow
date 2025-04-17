import gradio as gr
import imageio
import numpy as np
import onnx
import onnxruntime as rt
import time
from numpy.random import RandomState
from skimage import transform
import os

# ------------------------- Utility functions -------------------------

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
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    dw /= 2; dh /= 2
    if shape != new_unpad:
        im = transform.resize(im, (new_unpad[1], new_unpad[0]))
    top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
    left, right = int(round(dw-0.1)), int(round(dw+0.1))
    im_new = np.full((new_unpad[1]+top+bottom, new_unpad[0]+left+right, 3), color, dtype=np.float32)
    im_new[top:new_unpad[1]+top, left:new_unpad[0]+left] = im
    return im_new

def nms(pred, conf_thres, iou_thres, max_instance=20):
    nc = pred.shape[1] - 5
    candidates = [list() for _ in range(nc)]
    for x in pred:
        if x[4] < conf_thres:
            continue
        cls = np.argmax(x[5:])
        p = x[4] * x[5+cls]
        if p < conf_thres:
            continue
        box = (x[0]-x[2]/2, x[1]-x[3]/2, x[0]+x[2]/2, x[1]+x[3]/2)
        candidates[cls].append([p, box])
    result = [list() for _ in range(nc)]
    for i, cand in enumerate(candidates):
        cand = sorted(cand, key=lambda a: a[0], reverse=True)[:max_instance]
        for score, box in cand:
            if all(iou(box, kept[1]) <= iou_thres for kept in result[i]):
                result[i].append([score, box])
    return result

# --------------------------- Model wrapper ---------------------------

class Model:
    def __init__(self):
        self.load_models("./models/")
        # cache dimensions
        inputs_meta = self.g_mapping.get_inputs()
        self.z_dim   = inputs_meta[0].shape[-1]
        self.psi_dim = inputs_meta[1].shape[-1]
        self.gsyn_inputs = [inp.name for inp in self.g_synthesis.get_inputs()]

    def load_models(self, model_dir):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        try:
            gmap = onnx.load(model_dir + "g_mapping.onnx")
            w_avg = [x for x in gmap.graph.initializer if x.name == "w_avg"][0]
            w_avg = np.frombuffer(w_avg.raw_data, dtype=np.float32)[np.newaxis, :]
            w_avg = w_avg.repeat(16, axis=0)[np.newaxis, :]
        except Exception:
            w_avg = np.zeros((1, 16, 512), dtype=np.float32)
        self.w_avg = w_avg
        self.g_mapping   = rt.InferenceSession(model_dir + "g_mapping.onnx", providers=providers)
        self.g_synthesis = rt.InferenceSession(model_dir + "g_synthesis.onnx", providers=providers)

    # --- basic forward helpers ---
    def get_img(self, w):
        feed = {'w': w}
        if 'noise' in self.gsyn_inputs:
            feed['noise'] = np.array([1.0], dtype=np.float32)
        img = self.g_synthesis.run(None, feed)[0]
        return (img.transpose(0, 2, 3, 1) * 127.5 + 128).clip(0, 255).astype(np.uint8)[0]

    def get_w(self, z, psi):
        psi_input = np.full((self.psi_dim,), psi, dtype=np.float32)
        return self.g_mapping.run(None, {'z': z, 'psi': psi_input})[0]

    # --- video helpers ---
    def gen_video_pair(self, w1, w2, path, frame_num=10):
        writer = imageio.get_writer(path, mode='I', fps=frame_num // 2, codec='libx264', bitrate='16M')
        lin = np.linspace(0, 1, frame_num)
        for alpha in lin:
            img = self.get_img((1 - alpha) * w1 + alpha * w2)
            writer.append_data(img)
        writer.close()

    def gen_video_sequence(self, ws, path, frames_per_seg=15):
        writer = imageio.get_writer(path, mode='I', fps=frames_per_seg // 2, codec='libx264', bitrate='16M')
        lin = np.linspace(0, 1, frames_per_seg)
        for i in range(len(ws) - 1):
            for alpha in lin:
                img = self.get_img((1 - alpha) * ws[i] + alpha * ws[i + 1])
                writer.append_data(img)
        writer.close()

# --------------------------- Generation logic ---------------------------

def gen_img_fn(method, seed, psi):
    if method == 1:
        z = RandomState(int(seed) + 2**31).randn(1, model.z_dim)
    else:
        z = np.random.randn(1, model.z_dim)
    w = model.get_w(z.astype(np.float32), psi)
    img = model.get_img(w)
    filepath = f"outputs/images/seed_{seed}_psi_{psi:.2f}.png"
    imageio.imwrite(filepath, img)
    return img, w, img

def gen_video_fn(w1, w2, frames):
    if w1 is None or w2 is None:
        return None
    path = f"outputs/videos/pair_{int(time.time())}.mp4"
    model.gen_video_pair(w1, w2, path, int(frames))
    return path

# --- NEW: generate sequence of images and video automatically ---

def gen_sequence_fn(num_imgs, psi, frames_per_seg):
    num_imgs = int(num_imgs)
    if num_imgs < 2:
        return None, None
    seeds = np.random.randint(0, 2**31, size=num_imgs)
    ws, imgs = [], []
    for seed in seeds:
        z = RandomState(int(seed) + 2**31).randn(1, model.z_dim)
        w = model.get_w(z.astype(np.float32), psi)
        img = model.get_img(w)
        imgs.append(img)
        ws.append(w)
    video_path = f"outputs/videos/seq_{int(time.time())}.mp4"
    model.gen_video_sequence(ws, video_path, int(frames_per_seg))
    return imgs, video_path

# --------------------------- Gradio UI ---------------------------

os.makedirs("outputs/images", exist_ok=True)
os.makedirs("outputs/videos", exist_ok=True)

model = Model()
app = gr.Blocks()

with app:
    gr.Markdown("# full-body anime\n(the model is not well, just use for fun.)")
    with gr.Tabs():
                # --------------------- NEW: Sequence video tab ---------------------
        with gr.TabItem("generate sequence video"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## auto-generate N images and create a smooth sequence video")
                    seq_num_slider = gr.Slider(minimum=2, maximum=100, step=1, value=4, label="number of key images")
                    seq_psi_slider = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.6, label="truncation psi")
                    seq_frame_slider = gr.Slider(minimum=10, maximum=60, step=1, value=30, label="frames per transition")
                    seq_generate_button = gr.Button("Generate sequence video")
                with gr.Column():
                    seq_images_gallery = gr.Gallery(label="generated images").style(grid=[2], height="auto")
                    seq_video_output = gr.Video()

            seq_generate_button.click(
                gen_sequence_fn,
                [seq_num_slider, seq_psi_slider, seq_frame_slider],
                [seq_images_gallery, seq_video_output]
            )
        # --------------------- Image generation tab ---------------------
        with gr.TabItem("generate image"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("generate image randomly or by seed")
                    gen_input1 = gr.Radio(["random", "use seed"], label="method", type="index")
                    gen_input2 = gr.Number(1, label="seed")
                    gen_input3 = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.6, label="truncation psi")
                    gen_submit = gr.Button("Generate")
                with gr.Column():
                    gen_output1 = gr.Image()
                    select_img_input_w1 = gr.State()
                    select_img_input_img1 = gr.State()
            gen_submit.click(gen_img_fn,
                             [gen_input1, gen_input2, gen_input3],
                             [gen_output1, select_img_input_w1, select_img_input_img1])

        # --------------------- Pair video tab ---------------------
        with gr.TabItem("generate video"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## generate video between 2 images")
                    select_img1_dropdown = gr.Radio(["generated image", "encoded image"], label="source1", type="index")
                    select_img1_button = gr.Button("select")
                    select_img1_output_img = gr.Image(label="image 1")
                    select_img1_output_w = gr.State()

                    select_img2_dropdown = gr.Radio(["generated image", "encoded image"], label="source2", type="index")
                    select_img2_button = gr.Button("select")
                    select_img2_output_img = gr.Image(label="image 2")
                    select_img2_output_w = gr.State()

                    generate_video_frame = gr.Slider(minimum=10, maximum=30, step=1, value=15, label="frame")
                    generate_video_button = gr.Button("Generate")
                with gr.Column():
                    generate_video_output = gr.Video()

            select_img1_button.click(
                lambda i, img1, img2, w1, w2: (img1, w1) if i == 0 else (img2, w2),
                [select_img1_dropdown, gen_output1, gen_output1, select_img_input_w1, select_img_input_w1],
                [select_img1_output_img, select_img1_output_w]
            )
            select_img2_button.click(
                lambda i, img1, img2, w1, w2: (img1, w1) if i == 0 else (img2, w2),
                [select_img2_dropdown, gen_output1, gen_output1, select_img_input_w1, select_img_input_w1],
                [select_img2_output_img, select_img2_output_w]
            )
            generate_video_button.click(
                gen_video_fn,
                [select_img1_output_w, select_img2_output_w, generate_video_frame],
                [generate_video_output]
            )

app.launch(share=True)
