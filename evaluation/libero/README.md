# Evaluation on LIBERO

We evaluate **X-VLA** on the LIBERO benchmark, which consists of four subtasks: **Spatial**, **Object**, **Goal**, and **Long**.

---

## 1️⃣ Environment Setup

Set up LIBERO following the [official instructions](https://github.com/Lifelong-Robot-Learning/LIBERO).

```
conda create -n libero python=3.8.13
conda activate libero
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -e .
```

---

## 2️⃣ Start the X-VLA Server

Run the X-VLA model as an inference server (in a clean environment to avoid dependency conflicts):
```bash
cd X-VLA
conda activate X-VLA
python -m deploy \
  --model_path 2toINF/X-VLA-Libero \
  --port 8000
```

---

## 2.5️⃣ Single-Step Sample Inference

If you only want to sanity-check that the LIBERO checkpoint loads and produces an action plan, you can run a local forward pass without starting the server:

```bash
cd X-VLA
python sample_inference.py \
  --model_path /path/to/X-VLA-Libero \
  --image0 /path/to/agentview_rgb.png \
  --image1 /path/to/wrist_rgb.png \
  --instruction "pick up the white mug and place it in the bowl"
```

Notes:
- `sample_inference.py` defaults to `domain_id=3`, which is the LIBERO domain id used in this repo.
- If no images or proprio are passed, the script falls back to blank images and a zero 20-D proprio vector for a smoke test.

---


## 3️⃣ Run the Client Evaluation

Launch the LIBERO evaluation client to connect to your X-VLA server:
```bash
cd evaluation/libero
conda activate libero
python libero_client.py \
    --task_suites libero_spatial libero_goal libero_object libero_10 \
    --server_ip 0.0.0.0 \
    --server_port 8000
```


---
