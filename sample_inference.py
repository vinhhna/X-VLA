import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

model_id = "2toINF/X-VLA-Libero"
device = "cpu"
instruction = "pick up the object and place it in the box"
steps = 4

model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device).to(torch.float32)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

image = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
inputs = processor([image], instruction)

model_inputs = {}
for key, value in inputs.items():
    if isinstance(value, torch.Tensor):
        if value.is_floating_point():
            model_inputs[key] = value.to(device=device, dtype=torch.float32)
        else:
            model_inputs[key] = value.to(device=device)

proprio = torch.zeros((1, 20), dtype=torch.float32, device=device)
domain_id = torch.tensor([0], dtype=torch.long, device=device)

actions = model.generate_actions(
    input_ids=model_inputs["input_ids"],
    image_input=model_inputs["image_input"],
    image_mask=model_inputs["image_mask"],
    domain_id=domain_id,
    proprio=proprio,
    steps=steps,
)

actions_np = actions.squeeze(0).detach().cpu().numpy()
print("action shape:", actions_np.shape)
print("first row first 10:", np.array2string(actions_np[0, :10], precision=4))
